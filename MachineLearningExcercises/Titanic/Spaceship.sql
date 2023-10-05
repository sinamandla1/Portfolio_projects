--Query to check for Duplicates under 'Name' column
SELECT 
[Name]
FROM train
GROUP BY [Name]
HAVING COUNT(*) > 1;

--Tally True and False values of people who had VIP status
SELECT
  SUM(CASE WHEN VIP = 'True' THEN 1 ELSE 0 END) AS True_Count,
  SUM(CASE WHEN VIP = 'False' THEN 1 ELSE 0 END) AS False_Count
FROM train;

--Just to see how many home planets are in the data set
SELECT DISTINCT HomePlanet
FROM train;

--Tally of how many people from the distinct planets went to the most successful trip destination, to calculate travel success ratio
--Earth 3101, Mars 1475, Europa 1189.
SELECT
  SUM(CASE WHEN HomePlanet = 'Earth' AND Destination ='TRAPPIST-1e' THEN 1 ELSE 0 END) AS True_Count_Earth,
  SUM(CASE WHEN HomePlanet = 'Mars' AND Destination ='TRAPPIST-1e' THEN 1 ELSE 0 END) AS True_Count_Mars,
  SUM(CASE WHEN HomePlanet = 'Europa' AND Destination ='TRAPPIST-1e' THEN 1 ELSE 0 END) AS True_Count_Europa
FROM train;

--To see the entire record of each duplicate for analysis

SELECT DISTINCT *
FROM train
WHERE [Name] = 'Elaney Webstephrey';

SELECT DISTINCT *
FROM train
WHERE [Name] = 'Grake Porki';

SELECT DISTINCT *
FROM train
WHERE [Name] = 'Sus Coolez';

SELECT DISTINCT *
FROM train
WHERE [Name] = 'Apix Wala';

SELECT DISTINCT *
FROM train
WHERE [Name] = 'Sharie Gallenry';

SELECT DISTINCT *
FROM train
WHERE [Name] = 'Cuses Pread';

SELECT DISTINCT *
FROM train
WHERE [Name] = 'Alraium Disivering';

SELECT DISTINCT *
FROM train
WHERE [Name] = 'Carry Contrevins';

SELECT DISTINCT *
FROM train
WHERE [Name] = 'Keitha Josey';

SELECT DISTINCT *
FROM train
WHERE [Name] = 'Glenna Valezaley';

SELECT DISTINCT *
FROM train
WHERE [Name] = 'Dia Cartez';

SELECT DISTINCT *
FROM train
WHERE [Name] = 'Loree Wolfernan';

SELECT DISTINCT *
FROM train
WHERE [Name] = 'Gwendy Sykess';

SELECT DISTINCT *
FROM train
WHERE [Name] = 'Asch Stradick';

SELECT DISTINCT *
FROM train
WHERE [Name] = 'Troya Schwardson';

SELECT DISTINCT *
FROM train
WHERE [Name] = 'Glena Hahnstonsen';

SELECT DISTINCT *
FROM train
WHERE [Name] = 'Anton Woody';

SELECT DISTINCT *
FROM train
WHERE [Name] = 'Juane Popelazquez';

SELECT DISTINCT *
FROM train
WHERE [Name] = 'Ankalik Nateansive';
