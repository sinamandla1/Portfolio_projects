/*
--View the table and see how the data looks
SELECT TOP(10) *
FROM Content;

--Begin data cleaning

--View all values in the URL column where the record is NULL
SELECT *
FROM Content
WHERE [URL] is NULL;

--then delete all null values
DELETE FROM Content
WHERE [URL] is NULL;
*/

SELECT * FROM Content;

/*SELECT Top(10) *
FROM Reactions;

--DATA CLEANING
---------------------------------
--there is more than 1 column with null values, so there is a combination, we search until we get a query generated
--and then Delete them
SELECT *
FROM Reactions
WHERE Content_ID is NUll AND [User_ID] is null AND [Type] is Null and [Datetime] is NULL;

SELECT *
FROM Reactions
WHERE [User_ID] is null AND [Type] is Null;

DELETE FROM Reactions
WHERE [User_ID] is null AND [Type] is Null;
*/
-------------------------------------

--Some null values remain so we delete those too
/*SELECT *
FROM Reactions
WHERE [User_ID] is null;

DELETE FROM Reactions
WHERE [User_ID] is null;

---No null values anymore here
SELECT *
FROM Reactions
WHERE [Type] is null;

--CHeck the data
SELECT * FROM Reactions
WHERE [Type] is NULL OR [User_ID] is NULL;
*/

SELECT * FROM Reactions;

/*
SELECT Top(10) *
FROM ReactionTypes;

SELECT *
FROM ReactionTypes
WHERE [Type] is NUll OR Sentiment is NULL or Score is NULL;
--The data is clean and has no Null values
*/

SELECT * FROM ReactionTypes;
