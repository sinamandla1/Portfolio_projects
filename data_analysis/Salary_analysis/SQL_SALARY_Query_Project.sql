-- Complex salary data set --

-----SKILLS USED: View Tables, MAX Function, Aggregates, CASES, Temp tables and Converting data types.

--------------View Tables--------------------
--1.
--View table querying the structure to get the various Job Titles in the data set and the
--number of people(regardless of age) employed in each role.
CREATE VIEW [Job title count] AS
SELECT [Job Title],
  COUNT(*) AS 'Number of employees'
FROM Salary_Data_Based_country_and_r$
GROUP BY [Job Title];

--2.
--- View table querying the structure to see the gender is distributed in this data set. With this
---we can determine averages of employed gender distribution focusing on ages 35yrs and younger.
CREATE VIEW [Total Youth global workforce Males vs Females] AS
SELECT
 SUM(CASE WHEN Gender = 'male' THEN 1 ELSE 0 END) AS 'Total males aged 35 and younger at work',
 SUM(CASE WHEN Gender = 'female' THEN 1 ELSE 0 END) AS 'Total females aged 35 and younger at work'
FROM Salary_Data_Based_country_and_r$
WHERE TRY_CAST(Age AS FLOAT) <= 35.0;

--3.
---View table querying the structure to see the gender is distributed in this data set. With this
---we can get industry averages of employement of males and females.

SELECT
 SUM(CASE WHEN Gender = 'male' THEN 1 ELSE 0 END) AS 'Total Males Working',
 SUM(CASE WHEN Gender = 'female' THEN 1 ELSE 0 END) AS 'Total Females Working'
FROM Salary_Data_Based_country_and_r$;

--4.
---View table querying the structure to get the various Job Titles in the data set and the
---number of people aged 35yrs and younger employed in each role, We then .
CREATE VIEW [Youth in Job title] AS
SELECT [Job Title],
  COUNT(*) AS 'Number of employees 35 and younger'
FROM Salary_Data_Based_country_and_r$
WHERE TRY_CAST(Age AS FLOAT) <= 35.0
GROUP BY [Job Title];
---------------------------------------------------------------------------------
-----QUERIES-----
--1.
---Ranking each country according to how many people it employs. This would
---help people looking for employment to set the sights onto places with high
---numbers of employment to take advantage of the higher odds available.

SELECT Country,
  COUNT(*) AS 'Number of Employees'
FROM Salary_Data_Based_country_and_r$
GROUP BY Country
ORDER BY [Number of Employees] DESC;

--2.
---Ranking each country according to how many Software engineers it employs. 
---This would help people looking for employment to set the sights onto places
---with high numbers of employment to take advantage of the higher odds available.

SELECT Country,
  COUNT(*) AS 'Number of Software Engineers'
FROM Salary_Data_Based_country_and_r$
WHERE [Job Title] = 'Software Engineer'
GROUP BY Country
ORDER BY [Number of Software Engineers] DESC;

--3.
---Assessing the distribution of genders in the number of Software Engineers
---employed. This will show how inclusive the job role is. We also see how socio-economic
---and industry policies are performing.

SELECT [Job Title],
  SUM(CASE WHEN Gender = 'Male' THEN 1 ELSE 0 END) AS 'Number of Male Software Engineers',
  SUM(CASE WHEN Gender = 'Female' THEN 1 ELSE 0 END) AS 'Number of Female Software Engineers'
FROM Salary_Data_Based_country_and_r$
WHERE [Job Title] = 'Software Engineer'
GROUP BY [Job Title];

--4.
---Assessing the distribution of genders in the number of Software Engineers
---employed, aged 35yrs and younger. This will show how the youth is active the job role is.

SELECT [Job Title],
  SUM(CASE WHEN Gender = 'Male' THEN 1 ELSE 0 END) AS 'Number of Male Software Engineers 35 and younger',
  SUM(CASE WHEN Gender = 'Female' THEN 1 ELSE 0 END) AS 'Number of Female Software Engineers 35 and younger'
FROM Salary_Data_Based_country_and_r$
WHERE [Job Title] = 'Software Engineer' AND TRY_CAST(Age AS FLOAT) <= 35.0
GROUP BY [Job Title];

--5.
---Ranking the employment in each country according to the distribution of genders involved.

SELECT Country,
  SUM(CASE WHEN Gender = 'Male' AND [Job title] = 'Software Engineer' THEN 1 ELSE 0 END) AS 'Number of Male Software Engineers',
  SUM(CASE WHEN Gender = 'Female' AND [Job title] = 'Software Engineer' THEN 1 ELSE 0 END) AS 'Number of Female Software Engineers'
FROM Salary_Data_Based_country_and_r$
GROUP BY Country
ORDER BY 2,3 DESC;

--6.
---looking at the distribution of genders in the highest paying field in the data set,
---within a specific country. We can see how effective their socio-economic environment is.

SELECT Country,
  SUM(CASE WHEN Gender = 'Male' THEN 1 ELSE 0 END) AS 'Number of Male Software Engineers 35 and younger',
  SUM(CASE WHEN Gender = 'Female' THEN 1 ELSE 0 END) AS 'Number of Female Software Engineers 35 and younger'
FROM Salary_Data_Based_country_and_r$
WHERE [Job Title] = 'Software Engineer' AND TRY_CAST(Age AS FLOAT) <= 35.0
GROUP BY Country
ORDER BY 2 DESC;

--7.
---specifically looking at the highest paying job, we want to see how the picture
---looks racially and look to see whether empowerment programs are needed.

SELECT Race,
  SUM(CASE WHEN [Job Title] = 'Software Engineer' THEN 1 ELSE 0 END) AS 'Number of employees'
FROM Salary_Data_Based_country_and_r$
GROUP BY Race
ORDER BY [Number of employees] DESC;
--DONE!--
