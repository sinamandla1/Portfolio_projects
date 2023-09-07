--SKILLS USED: View table, JOINS and Sub queries
---View table--

CREATE VIEW StoresOverview AS
SELECT [state],
       COUNT(city) AS NumbOfStores
FROM Stores
GROUP BY [state];

------------------

-- We see that we have 16 different states and how many stores in each
SELECT [state],
COUNT(city) AS NumbOfStores FROM stores
GROUP BY [state]
ORDER BY NumbOfStores DESC;

--Created a View table called "StoresOverview" and checked how many outlets in total in Equador
SELECT COUNT(state) AS TotalStates, SUM(NumbOfStores) AS TotalOutlets FROM StoresOverview;

--number of cities:
SELECT SUM(B.CityCount) as TotalCities FROM 
(SELECT COUNT(A.NumbOfCities) AS CityCount FROM 
(SELECT DISTINCT city AS NumbOfCities FROM stores)A GROUP BY NumbOfCities)B GROUP BY CityCount;

-----JOINS-------
-- I joined the predicted values from the model I trained to the test data set
SELECT A.*, B.sales
FROM test A
INNER JOIN dbo.predictions$ B
ON A.id=B.id;

-----------------
---see the exact types of Locale holidays in 2017
SELECT DISTINCT locale
FROM holidays_events
WHERE [date] like '2017%';

---see the full range of holidays in 2017
SELECT DISTINCT *
FROM holidays_events
WHERE [date] like '2017%';

---Types of product range in the test data set 
SELECT DISTINCT [family]
FROM test;
