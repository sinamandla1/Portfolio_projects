-- Complex salary data set

-----SKILLS USED: CTE, MAX Function, CASES, Temp tables and Converting data types

---After importing the csv file to Excel and converting it into a table and cleaning
---Check if the query results correspond with the excel file table shape.

SELECT Gender,
  Age,
  Race,
  [Job Title],
  [Education Level],
  Country,
  [Years of Experience],
  Salary
FROM Salary_Data_Based_country_and_r$

---Query the structure to find out what roles youths (30 years and younger) are employed in,
---using DISTINCT to give cleaner results.

SELECT DISTINCT Gender,
  Age,
  Race,
  [Job Title],
  [Education Level],
  Country,
  [Years of Experience],
  Salary
FROM Salary_Data_Based_country_and_r$
WHERE TRY_CAST(Age AS FLOAT) <= 30.0
ORDER BY Age, [Years of Experience];

---Querying the structure to see the gender is distributed in this data set. With this
---we can assess the odds of getting employed as a specific gender aged 30yrs and younger,
---or to assess current industry policies.

SELECT
 SUM(CASE WHEN Gender = 'male' THEN 1 ELSE 0 END) AS 'Total males aged 30 and younger at work',
 SUM(CASE WHEN Gender = 'female' THEN 1 ELSE 0 END) AS 'Total females aged 30 and younger at work'
FROM Salary_Data_Based_country_and_r$
WHERE TRY_CAST(Age AS FLOAT) <= 30.0;

---Querying the structure to see the gender is distributed in this data set. With this
---we can assess the odds of getting employed as a specific gender, or to assess current
---industry or organisational policies.

SELECT
 SUM(CASE WHEN Gender = 'male' THEN 1 ELSE 0 END) AS 'Total Males Working',
 SUM(CASE WHEN Gender = 'female' THEN 1 ELSE 0 END) AS 'Total Females Working'
FROM Salary_Data_Based_country_and_r$

---Querying the structure to get the various Job Titles in the data set and the
---number of people employed in each role.

SELECT [Job Title],
  COUNT(*) AS 'Number of employees'
FROM Salary_Data_Based_country_and_r$
GROUP BY [Job Title]
ORDER BY [Number of employees] DESC;

---Querying the structure to get the various Job Titles in the data set and the
---number of people aged 30yrs and younger employed in each role.

SELECT [Job Title],
  COUNT(*) AS 'Number of employees 30 and younger'
FROM Salary_Data_Based_country_and_r$
WHERE TRY_CAST(Age AS FLOAT) <= 30.0
GROUP BY [Job Title]
ORDER BY [Number of employees 30 and younger] DESC;

---Ranking each country according to how many people it employs. This would
---help people looking for employment to set the sights onto places with high
---numbers of employment to take advantage of the higher odds available.

SELECT Country,
  COUNT(*) AS 'Number of employees'
FROM Salary_Data_Based_country_and_r$
GROUP BY Country
ORDER BY [Number of employees] DESC;

---Ranking each country according to how many Software engineers it employs. 
---This would help people looking for employment to set the sights onto places
---with high numbers of employment to take advantage of the higher odds available.

SELECT Country,
  COUNT(*) AS 'Number of Software Engineers'
FROM Salary_Data_Based_country_and_r$
WHERE [Job Title] = 'Software Engineer'
GROUP BY Country
ORDER BY [Number of Software Engineers] DESC;

---Assessing the distribution of genders in the number of Software Engineers
---employed. This will show how inclusive the job role is. We also see how socio-economic
---and industry policies are performing.

SELECT [Job Title],
  SUM(CASE WHEN Gender = 'Male' THEN 1 ELSE 0 END) AS 'Number of Male Software Engineers',
  SUM(CASE WHEN Gender = 'Female' THEN 1 ELSE 0 END) AS 'Number of Female Software Engineers'
FROM Salary_Data_Based_country_and_r$
WHERE [Job Title] = 'Software Engineer'
GROUP BY [Job Title];

---Assessing the distribution of genders in the number of Software Engineers
---employed, aged 30yrs and younger. This will show how inclusive the job role is.
---We also see how socio-economic and industry policies are performing.

SELECT [Job Title],
  SUM(CASE WHEN Gender = 'Male' THEN 1 ELSE 0 END) AS 'Number of Male Software Engineers 30 and younger',
  SUM(CASE WHEN Gender = 'Female' THEN 1 ELSE 0 END) AS 'Number of Female Software Engineers 30 and younger'
FROM Salary_Data_Based_country_and_r$
WHERE [Job Title] = 'Software Engineer' AND TRY_CAST(Age AS FLOAT) <= 30.0
GROUP BY [Job Title];

---Ranking the employment in each country according to the distribution of genders involved.

SELECT Country,
  SUM(CASE WHEN Gender = 'Male' THEN 1 ELSE 0 END) AS 'Number of Male Software Engineers',
  SUM(CASE WHEN Gender = 'Female' THEN 1 ELSE 0 END) AS 'Number of Female Software Engineers'
FROM Salary_Data_Based_country_and_r$
GROUP BY Country
ORDER BY 2,3 DESC;

---looking at the distribution of genders in the highest paying field in the data set,
---within a specific country. We can see how effective their socio-economic environment is.

SELECT Country,
  SUM(CASE WHEN Gender = 'Male' THEN 1 ELSE 0 END) AS 'Number of Male Software Engineers 30 and younger',
  SUM(CASE WHEN Gender = 'Female' THEN 1 ELSE 0 END) AS 'Number of Female Software Engineers 30 and younger'
FROM Salary_Data_Based_country_and_r$
WHERE [Job Title] = 'Software Engineer' AND TRY_CAST(Age AS FLOAT) <= 30.0
GROUP BY Country
ORDER BY 2 DESC;

---specifically looking at the highest paying job, we want to see how the picture
---looks racially and look to see whether empowerment programs are needed.

SELECT Race,
  SUM(CASE WHEN [Job Title] = 'Software Engineer' THEN 1 ELSE 0 END) AS 'Number of employees'
FROM Salary_Data_Based_country_and_r$
GROUP BY Race
ORDER BY [Number of employees] DESC;

---specifically looking at the highest paying job, we also want to see how the picture
---looks racially for the youth population.

SELECT Race,
  SUM(CASE WHEN [Job Title] = 'Software Engineer' THEN 1 ELSE 0 END) AS 'Number of employees 30 and under'
FROM Salary_Data_Based_country_and_r$
WHERE TRY_CAST(Age AS FLOAT) <= 30.0
GROUP BY Race
ORDER BY [Number of employees 30 and under] DESC;

---Ranking the Job roles according to the highest salary and most years of experience
---for career choice sake, we would use this to know which part of the market to work
---towards for a meaningful career.

SELECT DISTINCT [Job Title],
  max(Salary) AS 'Highest salary',
  max([Years of Experience]) AS 'Highest years of experience'
FROM Salary_Data_Based_country_and_r$
GROUP BY [Job Title]
ORDER BY [Highest salary] DESC;