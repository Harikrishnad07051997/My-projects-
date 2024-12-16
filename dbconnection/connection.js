const { Sequelize } = require("sequelize");
require("dotenv").config();

// Ensure all required environment variables are set
if (!process.env.DB || !process.env.USER || !process.env.PASSWORD || !process.env.SQL_PORT) {
    console.error("Missing one or more environment variables: DB, USER, PASSWORD, SQL_PORT");
    process.exit(1); // Exit the process if any env variable is missing
}

const sequelize = new Sequelize("splitfile", 'sa', "baluK@143", {
    port:50340,
    host: 'localhost', // or your database host
    dialect: 'mssql',  // Use mssql for Microsoft SQL Server
  
    logging: console.log, // Enable logging (optional)
});

module.exports = {
    sequelize
};
