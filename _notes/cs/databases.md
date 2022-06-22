---
layout: notes
title: databases
category: cs
---



- overview
  - notes from [here](https://www.oracle.com/database/what-is-database/)
  - a database is an organized collection of structured information, or data
  -  typically modeled in rows and columns in a series of tables
  - SQL is a programming language used by nearly all [relational databases](https://www.oracle.com/database/what-is-database/#relational) to query, manipulate, and define data, and to provide access control
  - types
    - **Relational databases.** Relational databases became dominant in the 1980s. Items in a relational database are organized as a set of tables with columns and rows. Relational database technology provides the most efficient and flexible way to access structured information.
    - **Object-oriented databases.** Information in an object-oriented database is represented in the form of objects, as in object-oriented programming.
    - **Distributed databases.** A distributed database consists of two or more files located in different sites. The database may be stored on multiple computers, located in the same physical location, or scattered over different networks.
    - **Data warehouses.** A central repository for data, a data warehouse is a type of database specifically designed for fast query and analysis.
    - **NoSQL databases.** A [NoSQL](https://www.oracle.com/database/nosql-cloud.html), or nonrelational database, allows unstructured and semistructured data to be stored and manipulated (in contrast to a relational database, which defines how all data inserted into the database must be composed). NoSQL databases grew popular as web applications became more common and more complex.
    - **Graph databases.** A graph database stores data in terms of entities and the relationships between entities.
    - **OLTP databases.** An OLTP database is a speedy, analytic database designed for large numbers of transactions performed by multiple users.
  - examples
    - MySQL - simplest
- sql
  - the major commands:   `SELECT`, `UPDATE`, `DELETE`, `INSERT`, `WHERE`
  - SQL keywords are NOT case sensitive (i.e. can write `select`)