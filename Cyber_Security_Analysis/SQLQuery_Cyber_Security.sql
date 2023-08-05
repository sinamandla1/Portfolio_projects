---1: Test if the data is really imported

SELECT *
FROM ['Balloon Race Data Breaches - LA$'];

---2: Choose the relevant columns and order 

SELECT
  organisation,
  [records lost],
  [year   ],
  sector,
  method,
  [data sensitivity],
  story
FROM ['Balloon Race Data Breaches - LA$']
ORDER BY [records lost] DESC;

---3: count how many sectors

SELECT DISTINCT
  COUNT(sector) AS [number of sectors affected]
FROM ['Balloon Race Data Breaches - LA$'];

---4: choose distinct sectors and

SELECT DISTINCT
  Sector,
  [data sensitivity]
FROM ['Balloon Race Data Breaches - LA$']
WHERE [data sensitivity] >= 3
ORDER BY [data sensitivity] DESC;

---4: then get the average losses for each sector

SELECT
  sector,
  AVG(CAST(REPLACE([records lost], ',', '') AS BIGINT)) AS average_loss
FROM ['Balloon Race Data Breaches - LA$']
GROUP BY sector
ORDER BY average_loss DESC;

---5: Lets look at the most popular methods and the years that they occurred and order by the frequency

SELECT
  [year   ],
  method,
  COUNT(*) AS frequency
FROM ['Balloon Race Data Breaches - LA$']
GROUP BY [year   ], method
ORDER BY frequency DESC;