import os
import time
import json
import re
import tempfile
from dotenv import load_dotenv
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from openai import OpenAI
import pyodbc  
from werkzeug.utils import secure_filename
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict
from apscheduler.schedulers.background import BackgroundScheduler
import logging
import uvicorn
import fitz
import base64
from io import BytesIO
import tiktoken
from pydantic import BaseModel
import httpx 
# Load environment variables from .env file
load_dotenv()
form_recognizer_endpoint = os.getenv("AZURE_OCR_ENDPOINT")
form_recognizer_key = os.getenv("AZURE_OCR_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY") 
print(openai_api_key )
# Initialize tiktoken encoder for GPT-3.5-turbo
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
# initialize azure clients
document_analysis_client = DocumentAnalysisClient(
    endpoint=form_recognizer_endpoint,
    credential=AzureKeyCredential(form_recognizer_key)
)
openai_client = OpenAI(api_key=openai_api_key)  
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Database connection configuration
DB_SERVER = os.getenv("DB_SERVER")
DB_DATABASE = os.getenv("DB_DATABASE")
DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_DRIVER = os.getenv("DB_DRIVER", "{ODBC Driver 18 for SQL Server}")
DB_PORT=os.getenv("DB_PORT")
class VerifiedDocument(BaseModel):
    title: str
    start_page: int
    end_page: int
class GenerateSplitAgainRequest(BaseModel):
    id: int  # Changed from str to int for SQL Server
    verified_documents: List[VerifiedDocument]
class TokenCounter:
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.document_tokens = {}
    
    def count_tokens(self, text: str) -> int:
        return len(encoding.encode(text))
    
    def add_document_tokens(self, id: str, input_tokens: int, output_tokens: int):
        if id not in self.document_tokens:
            self.document_tokens[id] = {"input_tokens": 0, "output_tokens": 0}
        
        self.document_tokens[id]["input_tokens"] += input_tokens
        self.document_tokens[id]["output_tokens"] += output_tokens
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
def initialize_app():
    global scheduler
    
    # Initialize database
    initialize_database()
    
    # Only create and start scheduler if it doesn't exist
    if scheduler is None:
        scheduler = BackgroundScheduler()
        scheduler.add_job(process_unprocessed_documents, 'interval', minutes=5)
        try:
            scheduler.start()
            print("Scheduler started successfully")
        except Exception as e:
            print(f"Error starting scheduler: {e}")


document_types={
  "Deed": ["Effective Date", "Grantor", "Grantee", "Dated Date", "Recorded Date", "Book and Page OR Instrument Number (If you have them all,show them all)", "Blank notes", "Waterfront", "Property Type"],
  "Tax Assessment": ["Blank notes", "Tax ID", "Parcel number", "Tax Year", "Tax Type", "Tax amount", "Due Date", "Due amount", "Land", "Building", "Total amount", "Other Exemption Type", "Code Area"],
  "Mortgage": ["Dated Date", "Recorded Date", "Book and Page OR Instrument Number (If you have them all,show them all)", "Blank notes", "Lender/Mortgagee//Beneficiary", "Borrower/Mortgagor", "Amount", "Open Ended Amount", "Mortgage Tax (New York only)", "County of Record"],
  "Deed of Trust": ["Dated Date", "Recorded Date", "Book and Page OR Instrument Number (If you have them all,show them all)", "Blank notes", "Total amount", "Lender/Mortgagee//Beneficiary", "Borrower/Mortgagor", "Amount", "Open Ended Amount", "County of Record", "Trustee (if Deed of Trust)", "Trustee Address", "Trustee State", "Trustee ZIP", "Trustee County"],
  "Security Deed": ["Dated Date", "Recorded Date", "Book and Page OR Instrument Number (If you have them all,show them all)", "Blank notes", "Lender/Mortgagee//Beneficiary", "Borrower/Mortgagor", "Amount", "Open Ended Amount"],
  "Assignment": ["Dated Date", "Recorded Date", "Book and Page OR Instrument Number (If you have them all,show them all)", "Blank notes", "Trustee County", "Assignor", "Assignee", "Executed By (New Trustee)"],
  "Substitution of Trustee": ["Dated Date", "Recorded Date", "Book and Page OR Instrument Number (If you have them all,show them all)", "Blank notes", "Trustee (if Deed of Trust)", "Trustee Address", "Trustee State", "Trustee ZIP", "Trustee County", "Executed By (New Trustee)"],
  "Appointment of Substitute Trustee": ["Dated Date", "Recorded Date", "Book and Page OR Instrument Number (If you have them all,show them all)", "Blank notes", "Trustee (if Deed of Trust)", "Trustee Address", "Trustee State", "Trustee ZIP", "Trustee County", "Assignor", "Assignee", "Executed By (New Trustee)"],
  "Appointment of Successor Trustee": ["Dated Date", "Recorded Date", "Book and Page OR Instrument Number (If you have them all,show them all)", "Blank notes", "Trustee (if Deed of Trust)", "Trustee Address", "Trustee State", "Trustee ZIP", "Trustee County", "Assignor", "Assignee", "Executed By (New Trustee)"],
  "CEMA": ["Dated Date", "Recorded Date", "Book and Page OR Instrument Number (If you have them all,show them all)", "Blank notes", "Lender/Mortgagee//Beneficiary", "Borrower/Mortgagor", "Amount", "Mortgage Tax (New York only)", "New amount (CEMA)"],
  "Modification Agreement": ["Dated Date", "Recorded Date", "Book and Page OR Instrument Number (If you have them all,show them all)", "Blank notes", "Mortgage Tax (New York only)", "By", "Between/and", "Executed By"],
  "Notice of Default": ["Dated Date", "Recorded Date", "Book and Page OR Instrument Number (If you have them all,show them all)", "Blank notes", "Executed By (New Trustee)"],
  "Notice of Default and Election to Sell": ["Dated Date", "Recorded Date", "Book and Page OR Instrument Number (If you have them all,show them all)", "Blank notes", "Executed By (New Trustee)"],
  "Subordination Agreement": ["Dated Date", "Recorded Date", "Book and Page OR Instrument Number (If you have them all,show them all)", "Blank notes", "Executed By (New Trustee)", "Executed By", "To"],
  "Notice of Seizure (Louisiana)": ["Dated Date", "Recorded Date", "Book and Page OR Instrument Number (If you have them all,show them all)", "Blank notes"],
  "Order of Notice (Massachusetts)": ["Dated Date", "Recorded Date", "Book and Page OR Instrument Number (If you have them all,show them all)", "Blank notes"],
  "Power of Attorney": ["Dated Date", "Recorded Date", "Book and Page OR Instrument Number (If you have them all,show them all)"],
  "Power of Attorney to Foreclose Mortgage (Arkansas)": ["Dated Date", "Recorded Date", "Book and Page OR Instrument Number (If you have them all,show them all)"],
  "Judgment": ["Dated Date", "Recorded Date", "Book and Page OR Instrument Number (If you have them all,show them all)", "Blank notes", "Amount", "Plaintiff", "Defendant", "Case#", "Executed By", "Court Name"],
  "Federal Lien": ["Recorded Date", "Book and Page OR Instrument Number (If you have them all,show them all)", "Blank notes", "Total amount", "Amount", "Defendant", "Case#"],
  "Notice of Trustee's Sale": ["Dated Date", "Recorded Date", "Book and Page OR Instrument Number (If you have them all,show them all)", "Blank notes", "Trustee (if Deed of Trust)", "Executed By (New Trustee)", "Sale Date", "Sale Time"],
  "Re-recorded Document": ["Grantee", "Dated Date", "Recorded Date", "Book and Page OR Instrument Number (If you have them all,show them all)", "Blank notes"],
  "State Tax Lien": ["Recorded Date", "Book and Page OR Instrument Number (If you have them all,show them all)", "Blank notes", "Amount", "Plaintiff", "Defendant", "Case#"],
  "UCC Financing Statement": ["Recorded Date", "Book and Page OR Instrument Number (If you have them all,show them all)", "Blank notes", "Plaintiff", "Defendant"],
  "Federal Tax Lien": ["Recorded Date", "Book and Page OR Instrument Number (If you have them all,show them all)", "Blank notes", "Amount", "Defendant", "Case#"],
  "HOA Lien": ["Recorded Date", "Book and Page OR Instrument Number (If you have them all,show them all)", "Blank notes", "Amount", "Plaintiff", "Defendant"],
  "Claim of Lien": ["Recorded Date", "Book and Page OR Instrument Number (If you have them all,show them all)", "Blank notes", "Amount", "Plaintiff", "Defendant"],
  "County Tax Lien": ["Recorded Date", "Book and Page OR Instrument Number (If you have them all,show them all)", "Blank notes", "Amount", "Plaintiff", "Defendant"],
  "Lis Pendens": ["Dated Date", "Recorded Date", "Book and Page OR Instrument Number (If you have them all,show them all)", "Blank notes", "Plaintiff", "Defendant", "Case#", "Court Name"],
  "Request for Notice": ["Dated Date", "Recorded Date", "Book and Page OR Instrument Number (If you have them all,show them all)", "Blank notes", "Party and Address"],
  "Easement": ["Blank notes"],
  "Homestead Declaration": ["Recorded Date", "Book and Page OR Instrument Number (If you have them all,show them all)", "Blank notes", "Executed By (New Trustee)"],
  "CC&R": ["Blank notes"],
  "Affidavit of Affixture": ["Blank notes"],
  "Unsecured Tax Lien": ["Recorded Date", "Book and Page OR Instrument Number (If you have them all,show them all)", "Blank notes", "Tax Year", "Amount"],
  "WRIT of Attachment": ["Recorded Date", "Book and Page OR Instrument Number (If you have them all,show them all)", "Blank notes", "Plaintiff", "Defendant", "Case#", "Court Name"],
  "Notices": ["Recorded Date", "Book and Page OR Instrument Number (If you have them all,show them all)", "Blank notes", "Executed By"],
  "Prior Deed": [
    "Name of Grantee",
    "Type of Deed",
    "Name of the Grantor",
    "Dated Date",
    "Date of recording",
    "Deed Book",
    "Page Number",
    "County"
  ],
  "Vesting": [
    "Vesting Name"
  ],
  "Legal": [
    "Legal"
  ],
  "Subordination": [
    "from",
    "to",
    "dated",
    "book",
    "page",
    "county"
  ],
  "Assignment of Mortgage": [
    "Assignment to",
    "Filed date",
    "Deed book",
    "page"
  ],
  "Plat Map": [
    "Plat book",
    "Page"
  ],
  "CCR": [
    "Deed Book",
    "Page"
  ],
  "Easements": [
    "From",
    "To",
    "Dated",
    "Book",
    "Page"
  ],
  "Right of Way": [
    "To info",
    "Dated",
    "Book",
    "page"
  ],
  "FIFA": [
    "Year",
    "Amount",
    "Book",
    "page"
  ],
  "Writ of Fieri Ficias": [
    "Plaintiff",
    "Defendants",
    "Amount of Lien",
    "Book",
    "Page",
    "Dated Date",
    "Recording Date"
  ],
  "Attorney Lien": [
    "Plaintiff name from Judgment",
    "Defendants name from Judgment",
    "Date of Judgment",
    "Case Number from Judgment",
    "Judgment Amount",
    "Lien Book from Judgment",
    "Page from Judgment"
  ],
  "County Taxes": [
    "Year to be paid",
    "Amount",
    "Due date",
    "Last year",
    "Amount tax paid",
    "Date of paid"
  ],
  "City Taxes": [
    "Year taxes due",
    "Name of City",
    "amount due",
    "Last year",
    "City name",
    "Amount",
    "Date of paid"
  ],
  "Assessment of Bona Fida Agricultural": [
    "UCC File NO.",
    "Name of Debtor",
    "Name of Secured Party",
    "Filed Dated",
    "Filed Time",
    "Recording Date",
    "Expiring Date",
    "Deed Book",
    "Deed Page"
  ]
}
def get_db_connection():
    """
    Create a database connection with proper server and port configuration
    """
    try:
        # Build connection string - removed PORT as it's included in SERVER
        conn_str = (
            f'DRIVER={DB_DRIVER};'
            f'SERVER={DB_SERVER};'  # Server name should include port if needed
            f'DATABASE={DB_DATABASE};'
            f'UID={DB_USERNAME};'
            f'PWD={DB_PASSWORD};'
            'TrustServerCertificate=yes;'
            'Encrypt=yes;'
            'Connection Timeout=30'
        )
        
        conn = pyodbc.connect(conn_str)
        return conn

    except pyodbc.Error as e:
        error_message = f"Error connecting to database: {str(e)}"
        print(error_message)
        raise Exception(error_message)
def test_sql_connection():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Test the connection
        cursor.execute("SELECT @@VERSION")
        row = cursor.fetchone()
        print(f"SQL Server Version: {row[0]}")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"Connection test failed: {str(e)}")
        return False
# Create necessary tables if they don't exist
def initialize_database():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # First check if documents table exists
        cursor.execute("""
        IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'documents')
        BEGIN
            CREATE TABLE documents (
                id INT IDENTITY(1,1) PRIMARY KEY,
                image NVARCHAR(MAX),
                status NVARCHAR(50) DEFAULT 'notprocessed',
                error_message NVARCHAR(MAX),
                createdAt DATETIME DEFAULT GETDATE(),
                updatedAt DATETIME DEFAULT GETDATE()
            )
        END
        """)
        
        # Create imagepages table
        cursor.execute("""
        IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'imagepages')
        BEGIN
            CREATE TABLE imagepages (
                id INT IDENTITY(1,1) PRIMARY KEY,
                document_id INT,
                page_number INT,
                image_url NVARCHAR(MAX),
                createdAt DATETIME DEFAULT GETDATE(),
                updatedAt DATETIME DEFAULT GETDATE(),
                FOREIGN KEY (document_id) REFERENCES documents(id)
            )
        END
        """)
        
        # Create ocr_result table with NOT NULL constraints and default values
        cursor.execute("""
        IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'ocr_result')
        BEGIN
            CREATE TABLE ocr_result (
                id INT IDENTITY(1,1) PRIMARY KEY,
                document_id INT NOT NULL,
                page_number INT NOT NULL,
                original_content NVARCHAR(MAX),
                corrected_content NVARCHAR(MAX),
                createdAt DATETIME NOT NULL DEFAULT GETDATE(),
                updatedAt DATETIME NOT NULL DEFAULT GETDATE(),
                FOREIGN KEY (document_id) REFERENCES documents(id)
            )
        END
        """)

        # Create split table
        cursor.execute("""
        IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'split')
        BEGIN
            CREATE TABLE split (
                id INT IDENTITY(1,1) PRIMARY KEY,
                document_id INT NOT NULL,
                title NVARCHAR(MAX),
                start_page INT,
                end_page INT,
                content NVARCHAR(MAX),
                document_type NVARCHAR(100),
                extracted_fields NVARCHAR(MAX),
                createdAt DATETIME NOT NULL DEFAULT GETDATE(),
                updatedAt DATETIME NOT NULL DEFAULT GETDATE(),
                FOREIGN KEY (document_id) REFERENCES documents(id)
            )
        END
        """)
        
        conn.commit()
        print("Database initialized successfully")
        
    except Exception as e:
        print(f"Error initializing database: {e}")
        raise
    finally:
        if conn:
            conn.close()
def chunk_text(text, max_chunk_size=3000):
    """Split text into chunks of maximum token size"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        word_length = len(encoding.encode(word))
        if current_length + word_length > max_chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

def post_process_with_gpt(page_content, page_number, previous_page_content=None, id=None):
    print(f"Sending extracted text from page {page_number} to GPT for post-processing and document splitting...")

    try:
        system_message = """You are an AI assistant trained to process and analyze document content. 
        For each page, you must:
        1. Clean and format the text
        2. Determine if this is the start of a new document
        3. If it's a new document, identify its title
        
        Your response should be valid JSON with exactly these fields:
        {
            "corrected_content": "cleaned and formatted text",
            "is_new_document": true/false,
            "document_title": "title if new document, null if not"
        }"""
        
        # Chunk the content if it's too large
        max_tokens = 6000  # Leave room for system message and completion
        content_chunks = chunk_text(page_content, max_tokens)
        
        if len(content_chunks) > 1:
            # Process first chunk for document identification
            processed_result = process_chunk(content_chunks[0], system_message, page_number, previous_page_content, id)
            
            # Process remaining chunks just for content
            full_content = processed_result["corrected_content"]
            for chunk in content_chunks[1:]:
                chunk_result = process_chunk(chunk, system_message, page_number, None, id)
                full_content += "\n" + chunk_result["corrected_content"]
            
            processed_result["corrected_content"] = full_content
            return processed_result
        else:
            return process_chunk(page_content, system_message, page_number, previous_page_content, id)

    except Exception as e:
        print(f"Error during GPT processing for page {page_number}: {e}")
        return {
            "corrected_content": page_content,
            "is_new_document": False,
            "document_title": None
        }

def process_chunk(content, system_message, page_number, previous_page_content=None, id=None):
    user_message = f"""Process this text from page {page_number}:

Previous page content:
{previous_page_content if previous_page_content else "N/A"}

Current page content:
{content}

Return your response as a JSON object."""

    # Count input tokens
    input_tokens = token_counter.count_tokens(system_message + user_message)
    
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        temperature=0.3,
        max_tokens=2000
    )
    
    # Count output tokens
    output_tokens = token_counter.count_tokens(response.choices[0].message.content)
    
    # Add tokens to counter
    if id:
        token_counter.add_document_tokens(id, input_tokens, output_tokens)
    
    try:
        result = json.loads(response.choices[0].message.content)
        required_fields = ["corrected_content", "is_new_document", "document_title"]
        missing_fields = [field for field in required_fields if field not in result]
        
        if missing_fields:
            print(f"Missing required fields: {missing_fields}")
            return {
                "corrected_content": content,
                "is_new_document": False,
                "document_title": None
            }
        
        return result

    except json.JSONDecodeError as json_error:
        print(f"JSON parsing error for page {page_number}: {json_error}")
        print(f"Raw response: {response.choices[0].message.content}")
        return {
            "corrected_content": content,
            "is_new_document": False,
            "document_title": None
        }
# Modified process_document function with proper pdf_document handling
def process_document(id):
    conn = None
    cursor = None
    temp_file_path = None
    pdf_document = None
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Update status to processing with timestamp
        cursor.execute("""
            UPDATE documents 
            SET status = 'processing',
                error_message = NULL,
                updatedAt = GETDATE()
            WHERE id = ?
        """, id)
        conn.commit()
        
        # Get document info
        cursor.execute("SELECT image FROM documents WHERE id = ?", id)
        doc = cursor.fetchone()
        
        if not doc:
            raise Exception("Document not found")
        
        # Create temp file with unique name
        temp_file_path = os.path.join(tempfile.gettempdir(), f"doc_{id}_{int(time.time())}.pdf")
        
        # Download and process PDF
        response = httpx.get(doc[0])
        response.raise_for_status()
        
        with open(temp_file_path, 'wb') as file:
            file.write(response.content)
        
        # Process PDF pages to base64
        pdf_document = fitz.open(temp_file_path)
        
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_data = pix.tobytes("png")
            base64_string = base64.b64encode(img_data).decode()
            
            cursor.execute("""
                INSERT INTO imagepages (document_id, page_number, image_url, createdAt, updatedAt)
                VALUES (?, ?, ?, GETDATE(), GETDATE())
            """, (id, page_num + 1, base64_string))
        
        # Process PDF for OCR
        results, split_docs, token_usage = process_pdf(temp_file_path)
        
        # Store OCR results
        for page_num, result in results.items():
            cursor.execute("""
                INSERT INTO ocr_result 
                (document_id, page_number, original_content, corrected_content, createdAt, updatedAt)
                VALUES (?, ?, ?, ?, GETDATE(), GETDATE())
            """, (id, page_num, result['original_content'], result['corrected_content']))
        
        # Store split documents
        for doc in split_docs:
            cursor.execute("""
                INSERT INTO split 
                (document_id, title, start_page, end_page, content, document_type, extracted_fields, createdAt, updatedAt)
                VALUES (?, ?, ?, ?, ?, ?, ?, GETDATE(), GETDATE())
            """, (
                id,
                doc['title'],
                doc['start_page'],
                doc['end_page'],
                doc['content'],
                doc.get('document_type', 'Unknown'),
                json.dumps(doc.get('extracted_fields', {}))
            ))
        
        # Update document status to processed
        cursor.execute("""
            UPDATE documents 
            SET status = 'processed',
                error_message = NULL,
                updatedAt = GETDATE()
            WHERE id = ?
        """, id)
        
        conn.commit()
        
    except Exception as e:
        error_message = str(e)
        print(f"Error processing document {id}: {error_message}")
        if cursor and conn:
            cursor.execute("""
                UPDATE documents 
                SET status = 'failed', 
                    error_message = ?,
                    updatedAt = GETDATE()
                WHERE id = ?
            """, (error_message, id))
            conn.commit()
        raise
        
    finally:
        if pdf_document:
            pdf_document.close()
        if cursor:
            cursor.close()
        if conn:
            conn.close()
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as e:
                print(f"Error removing temp file: {e}")

def get_document_status(id):
    """
    Get the current status of a document
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT status, error_message 
            FROM documents 
            WHERE id = ?
        """, (id,))
        
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if result:
            return {
                'status': result[0],
                'error_message': result[1]
            }
        return None
        
    except Exception as e:
        print(f"Error getting document status: {e}")
        return None
def process_pdf(pdf_path):
    # Reset token counter for new document
    global token_counter
    token_counter = TokenCounter()
    
    ocr_results = extract_text_with_prebuilt_read(pdf_path)
    
    if not ocr_results:
        print("Error occurred during text extraction, aborting process.")
        return {}, [], {"input_tokens": 0, "output_tokens": 0}

    processed_results = {}
    documents = []
    current_document = None
    previous_page_content = None

    total_pages = len(ocr_results)
    id = f"pdf_{time.time()}"

    for page_number in range(1, total_pages + 1):
        original_content = ocr_results[str(page_number)]
        processed_page = post_process_with_gpt(original_content, page_number, previous_page_content, id)

        # Process pages as before...
        corrected_content = processed_page["corrected_content"]
        if isinstance(corrected_content, dict):
            corrected_content = str(corrected_content)

        processed_results[str(page_number)] = {
            "original_content": original_content,
            "corrected_content": corrected_content
        }

        if processed_page["is_new_document"] or current_document is None:
            if current_document:
                if isinstance(current_document["content"], list):
                    current_document["content"] = "\n".join(current_document["content"])
                documents.append(current_document)
            
            current_document = {
                "title": processed_page["document_title"] or f"Document starting on page {page_number}",
                "start_page": page_number,
                "end_page": page_number,
                "content": corrected_content
            }
        else:
            if isinstance(current_document["content"], str):
                current_document["content"] += f"\n\nPage {page_number}:\n{corrected_content}"
            else:
                current_document["content"] = f"{current_document['content']}\n\nPage {page_number}:\n{corrected_content}"
            current_document["end_page"] = page_number

        previous_page_content = corrected_content
        time.sleep(0.5)

    if current_document:
        if isinstance(current_document["content"], list):
            current_document["content"] = "\n".join(current_document["content"])
        documents.append(current_document)

    # Process documents and extract fields as before...
    for document in documents:
        document_type = get_matching_document_type(document["title"])
        document["document_type"] = document_type
        if not isinstance(document["content"], str):
            document["content"] = str(document["content"])
        document["extracted_fields"] = extract_fields(document["content"], document_type)

    # Add token usage information to the response
    token_usage = {
        "total_input_tokens": token_counter.total_input_tokens,
        "total_output_tokens": token_counter.total_output_tokens,
        "document_tokens": token_counter.document_tokens
    }

    return processed_results, documents, token_usage
def get_matching_document_type(title):
    if not title:
        return "Unknown"
        
    # convert title to lowercase and remove common words
    cleaned_title = re.sub(r'\b(of|the|a|an)\b', '', title.lower())
    cleaned_title = re.sub(r'\s+', ' ', cleaned_title).strip()
    
    # First try exact match
    for doc_type in document_types.keys():
        if doc_type.lower() == cleaned_title:
            return doc_type
    
    # Then try contains match
    for doc_type in document_types.keys():
        if doc_type.lower() in cleaned_title:
            return doc_type
    
    # Then try word match
    for doc_type in document_types.keys():
        doc_words = set(doc_type.lower().split())
        title_words = set(cleaned_title.split())
        if len(doc_words.intersection(title_words)) > 0:
            return doc_type
            
    # If still no match, try fuzzy matching
    highest_ratio = 0
    best_match = None
    
    for doc_type in document_types.keys():
        # Simple ratio comparison of strings
        ratio = sum(a == b for a, b in zip(doc_type.lower(), cleaned_title)) / max(len(doc_type), len(cleaned_title))
        if ratio > highest_ratio and ratio > 0.5:  # 50% threshold
            highest_ratio = ratio
            best_match = doc_type
    
    return best_match if best_match else "Other"  # Changed from "Unknown" to "Other"
def get_unprocessed_documents():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, image, status 
            FROM documents 
            WHERE status = 'notprocessed' 
            OR status IS NULL
        """)
        
        results = cursor.fetchall()
        print("power",results)
        unprocessed_docs = []
        
        for row in results:
            id = row[0]
            image = row[1]
            status = row[2] or 'notprocessed'
            
            if id is not None and image is not None:
                unprocessed_docs.append((id, image, status))
        
        return unprocessed_docs
        
    except Exception as e:
        print(f"Error fetching unprocessed documents: {e}")
        return []
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def process_unprocessed_documents():
    print("Processing unprocessed documents...")
    try:
        unprocessed_docs = get_unprocessed_documents()
        print(f"Found {len(unprocessed_docs)} unprocessed documents")
        
        for id, image, status in unprocessed_docs:
            try:
                print(f"Processing document ID: {id}")
                process_document(id)
                print(f"Successfully processed document ID: {id}")
                
            except Exception as e:
                error_message = str(e)
                print(f"Error processing document ID {id}: {error_message}")
                
                # Create a new connection for error handling
                error_conn = get_db_connection()
                try:
                    error_cursor = error_conn.cursor()
                    error_cursor.execute("""
                        UPDATE documents 
                        SET status = 'failed', 
                            error_message = ? 
                        WHERE id = ?
                    """, (error_message[:500], id))  # Limit error message length
                    error_conn.commit()
                finally:
                    error_conn.close()

    except Exception as e:
        print(f"Error in process_unprocessed_documents: {e}")
def extract_text_with_prebuilt_read(pdf_path):
    print(f"Submitting entire PDF for analysis")

    try:
        with open(pdf_path, "rb") as pdf_file:
            poller = document_analysis_client.begin_analyze_document("prebuilt-read", document=pdf_file)
            result = poller.result()

            extracted_text = {}

            for i, page in enumerate(result.pages, start=1):
                page_content = ""
                for line in page.lines:
                    page_content += line.content + "\n"

                extracted_text[str(i)] = page_content.strip()

            print(f"Text extraction completed for the entire PDF.")

        return extracted_text

    except Exception as e:
        print(f"Error during text extraction: {e}")
        return {}

def extract_fields(content, document_type):
    if document_type not in document_types:
        print(f"Unknown document type: {document_type}")
        return {}

    fields_to_extract = document_types[document_type]
    
    try:
        system_message = """You are an AI assistant trained to extract specific fields from legal documents. 
        Your response should be a JSON object containing the extracted fields.
        If a field is not found, set its value to null."""
        
        user_message = f"""Extract these fields from the {document_type} document:
        {', '.join(fields_to_extract)}

        Document content:
        {content}

        Return your response as a JSON object with the field names as keys."""

        input_tokens = token_counter.count_tokens(system_message + user_message)
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.2,
            max_tokens=1000
        )

        output_tokens = token_counter.count_tokens(response.choices[0].message.content)
        
        id = f"doc_{document_type}_{time.time()}"
        token_counter.add_document_tokens(id, input_tokens, output_tokens)

        try:
            extracted_fields = json.loads(response.choices[0].message.content)
            return extracted_fields
        except json.JSONDecodeError as e:
            print(f"JSON parsing error in field extraction: {e}")
            print(f"Raw response: {response.choices[0].message.content}")
            return {field: None for field in fields_to_extract}

    except Exception as e:
        print(f"Error during field extraction: {e}")
        return {field: None for field in fields_to_extract}
@app.get("/document_status/{id}")
async def document_status(id: int):
    status = get_document_status(id)
    if status:
        return JSONResponse(content=status)
    raise HTTPException(status_code=404, detail="Document not found")
@app.post("/generate_split_again")
async def generate_split_again(request: GenerateSplitAgainRequest):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Get document and OCR results
        cursor.execute("""
        SELECT d.*, o.page_number, o.corrected_content 
        FROM documents d
        LEFT JOIN ocr_result o ON d.id = o.document_id
        WHERE d.id = ?
        """, (request.id,))
        
        results = cursor.fetchall()
        if not results:
            raise HTTPException(status_code=404, detail="Document not found")

        # Update document status
        cursor.execute("UPDATE documents SET status = 'processing' WHERE id = ?", (request.id,))
        
        # Process verified documents
        for verified_doc in request.verified_documents:
            content = ""
            for page_num in range(verified_doc.start_page, verified_doc.end_page + 1):
                cursor.execute("""
                SELECT corrected_content 
                FROM ocr_result 
                WHERE document_id = ? AND page_number = ?
                """, (request.id, page_num))
                
                result = cursor.fetchone()
                if result:
                    content += f"\nPage {page_num}:\n{result.corrected_content}"

            document_type = get_matching_document_type(verified_doc.title)
            extracted_fields = extract_fields(content, document_type)

            # Update split documents
            cursor.execute("""
            INSERT INTO split
            (document_id, title, start_page, end_page, content, document_type, extracted_fields)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                request.id,
                verified_doc.title,
                verified_doc.start_page,
                verified_doc.end_page,
                content.strip(),
                document_type,
                json.dumps(extracted_fields)
            ))

        # Update document status
        cursor.execute("UPDATE documents SET status = 'processed' WHERE id = ?", (request.id,))
        conn.commit()

        # Get updated split documents
        cursor.execute("""
        SELECT * FROM split WHERE document_id = ?
        """, (request.id,))
        updated_documents = cursor.fetchall()

        return {"message": "Document split updated successfully", "updated_documents": updated_documents}

    except Exception as e:
        print(f"Error in generate_split_again: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

def process_documents():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get documents that need processing
        cursor.execute("""
        SELECT TOP 10 id 
        FROM documents 
        WHERE status IN ('processing', 'notprocessed')
        """)
        
        documents = cursor.fetchall()
        
        for doc in documents:
            try:
                process_document(doc.id)
                time.sleep(1)
            except Exception as e:
                logging.error(f"Error processing document {doc.id}: {e}")
                cursor.execute("""
                UPDATE documents 
                SET status = 'failed', error_message = ? 
                WHERE id = ?
                """, (str(e), doc.id))
                conn.commit()
    
    finally:
        conn.close()

# Initialize scheduler
scheduler = BackgroundScheduler()
print("cron job")
scheduler.add_job(process_unprocessed_documents, 'interval', minutes=1)

@app.on_event("startup")
async def startup_event():
    try:
        initialize_database()
        scheduler.start()
        print("Application started successfully")
    except Exception as e:
        print(f"Error during startup: {e}")
        raise
@app.on_event("shutdown")
async def shutdown_event():
    global scheduler
    if scheduler:
        scheduler.shutdown()
        scheduler = None
        print("Scheduler shut down successfully")

# Modified main block
if __name__ == "__main__":
    # Don't start the scheduler here anymore
    uvicorn.run(app, host="0.0.0.0", port=8000)