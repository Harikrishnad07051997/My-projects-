require('dotenv').config();
const { MongoClient } = require('mongodb');

// Ensure that the MONGO_URI and DB_NAME environment variables are set
const url = process.env.MONGO_URI;
console.log("url",url)
const dbName = process.env.MONGO_BUCKET_NAME; // Assuming you have a DB_NAME variable

if (!url) {
    console.error('MONGO_URI environment variable is missing!');
    process.exit(1); // Exit if MONGO_URI is not defined
}

if (!dbName) {
    console.error('DB_NAME environment variable is missing!');
    process.exit(1); // Exit if DB_NAME is not defined
}

async function connectToMongoDB() {
    const client = new MongoClient(url);

    try {
        // Connect to the MongoDB cluster
        await client.connect();
        console.log('Connected to MongoDB');

        // Access the database
        const db = client.db(dbName);

        const collection = db.collection('credentials2');
        const document = await collection.findOne();
        console.log('document.credentials :>> ', document.credentials);
        return document.credentials;

        

    } catch (err) {
        console.error('Error connecting to MongoDB:', err);
    } finally {
        // Close the connection
        await client.close();
    }
}

// Export the function to allow for use elsewhere in your application
module.exports = connectToMongoDB;
