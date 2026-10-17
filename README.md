# SSMU-Data-Challenge-I

Welcome to Data Challenge I! You and your teammates will have from 11 am to 5:30 pm on Saturday, October 17th, to explore the data and submit a presentation on your analysis.

This Git repository contains this `README` file and the dataset in a `CSV` format. If you are a Duke student, we recommend you use Duke's Container Manager to access your preferred IDE.

## Our Analysis

| Path | What it is |
| --- | --- |
| `newest/nyc_crash_graphs.ipynb` | Main notebook: COVID, Vision Zero, April 2020 reporting change, 34th Avenue Open Street, and seasonality graphs (`newest/graphs/`, tables in `newest/data/`). |
| `analysis/build-panels.py` | Downloads and verifies the raw archives, then builds the daily/monthly crash panels in `data/processed/` that everything else reads. |
| `analysis/covid-statistics.py` | Segmented regression and injury-share z test around the March 2020 COVID break → `outputs/covid_statistics/`. |
| `analysis/vision-zero-statistics.py` | Segmented regression of injuries and deaths around Vision Zero → `outputs/vision_zero_statistics/`. |
| `analysis/34ave-statistics.py` | Difference-in-differences of 34th Avenue against parallel avenues → `outputs/34ave_statistics/`. |
| `src/` | Standalone visualizations: traffic-adjusted time patterns, injury heatmap, monthly crash animations → `outputs/`. |

Run from the repository root:

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python analysis/build-panels.py        # first: data download + panels
python analysis/covid-statistics.py
python analysis/vision-zero-statistics.py
python analysis/34ave-statistics.py
```

Then open `newest/nyc_crash_graphs.ipynb` and run all cells. Results describe police-reported crashes. The April 2020 reporting change, shifting travel volumes, and the lack of randomized comparisons all limit causal claims.

## How to Participate

1. Fork the repository emailed to you by susan.li@duke.edu
2. Load the data from `motor_vehicle_collisions.zip`. Click the file, click “View Raw,” and download it.
3. Come up with a question to answer with data analysis. Analyze the dataset and prepare a presentation of your findings.
4. Submit your files to your team’s Google Drive folder. Submission materials should include: your code notebook file, presentation media, and citations to all outside resources.
5. Present! **You must fit your presentation within a 5-minute timeframe**. Please do not push changes to this repository; it is read-only for participants.
The analysis period will run from ~11:30 am to 4:30 pm. You and your team may stay in Physics 150 or leave during the analysis period (to get food, analyze in another room, etc.). You and your team must return to Physics 150 by 4:30 pm sharp to submit your presentation materials.

## Data Dictionary

**About**

You will be working with NYC motor vehicle collision data from data.gov. The Motor Vehicle Collisions data tables contain information from all police-reported motor vehicle collisions in NYC. Each row represents a crash event. The police report (MV104-AN) is required to be filled out for collisions where someone is injured or killed, or where there is at least $1000 worth of damage. The dataset includes 2,269,187 records across 29 columns.

Data Source: City of New York

Last Updated: July 31, 2026

Link: https://catalog.data.gov/dataset/motor-vehicle-collisions-crashes
We provided a zip file with the CSV for your team's convenience, but you can also download the other available files from the website.

**Data Variables**

| Column | Data type | Description |
| --- | --- | --- |
| CRASH DATE | character | Date the crash occurred. |
| CRASH TIME | double | Time the crash occurred. |
| BOROUGH | character | NYC borough where the crash occurred. |
| ZIP CODE | double | ZIP code where the crash occurred. |
| LATITUDE | double | Latitude in decimal degrees, using WGS 1984 (EPSG:4326). |
| LONGITUDE | double | Longitude in decimal degrees, using WGS 1984 (EPSG:4326). |
| LOCATION | character | Combined latitude and longitude. |
| ON STREET NAME | character | Street where the crash occurred. |
| CROSS STREET NAME | character | Nearest cross street to the crash. |
| OFF STREET NAME | character | Street address, if known. |
| NUMBER OF PERSONS INJURED | double | Total number of people injured. |
| NUMBER OF PERSONS KILLED | double | Total number of people killed. |
| NUMBER OF PEDESTRIANS INJURED | double | Number of pedestrians injured. |
| NUMBER OF PEDESTRIANS KILLED | double | Number of pedestrians killed. |
| NUMBER OF CYCLIST INJURED | double | Number of cyclists injured. |
| NUMBER OF CYCLIST KILLED | double | Number of cyclists killed. |
| NUMBER OF MOTORIST INJURED | double | Number of vehicle occupants injured. |
| NUMBER OF MOTORIST KILLED | double | Number of vehicle occupants killed. |
| CONTRIBUTING FACTOR VEHICLE 1 | character | Reported contributing factor for vehicle 1. |
| CONTRIBUTING FACTOR VEHICLE 2 | character | Reported contributing factor for vehicle 2. |
| CONTRIBUTING FACTOR VEHICLE 3 | character | Reported contributing factor for vehicle 3. |
| CONTRIBUTING FACTOR VEHICLE 4 | character | Reported contributing factor for vehicle 4. |
| CONTRIBUTING FACTOR VEHICLE 5 | logical | Reported contributing factor for vehicle 5. |
| COLLISION_ID | double | Unique crash identifier; primary key of the crash table. |
| VEHICLE TYPE CODE 1 | character | Type of vehicle 1. |
| VEHICLE TYPE CODE 2 | character | Type of vehicle 2. |
| VEHICLE TYPE CODE 3 | character | Type of vehicle 3. |
| VEHICLE TYPE CODE 4 | character | Type of vehicle 4. |
| VEHICLE TYPE CODE 5 | logical | Type of vehicle 5. |


# Competition Policies
## Code of Conduct
### AI and Online Sources
* You may not put this data into an LLM. 
* Analysis in this competition should reflect your own work; although we cannot prohibit AI, you should not depend on it in your analysis. Obvious or extreme LLM use will be considered in judging.
All citations must be listed at the end of your presentation media:
* Your team must include citations for forums (Stack Overflow), textbooks, and other references.
* You must provide a link to any AI chat conversations used.
* You must list all citations at the end of your presentation media. This can be a link to the data or an acknowledgment of the data used. 
* You may reuse code from your own research or class projects.

## Analysis
* You must answer a question that your team chooses.
* All types of data modeling are allowed.
* All coding languages are allowed.
* If previous research has been done on this topic, you may not copy the study.

## Submission Requirements
You and your team should submit the following to your Google Drive folder:
* Your code notebook file(s) (QMD, R script, Jupyter, etc.)
* Presentation media (slides, PDF, link to an HTML, visualizations, etc.).
* Citations to all outside resources: please read the Code of Conduct for more information.
* A link to your forked GitHub repository at the end of your presentation media.
* Your team’s Google Drive folder contents will not be accessible to view, comment on, or edit by any other team.

## Judging
* Winners will be announced at the end of the competition.
* As stated earlier, you must come up with a question that your team will use to guide your statistical analysis. 
* A fair evaluation will be conducted by the Statistical Science Majors Union's Competitions and Opportunities Committee.
* The judging criteria are as follows:
1) Communication: Clarity of presentation, logical structure, and ability to explain findings to a broad audience. - 20%
2) Creativity & Insight: Depth of analysis, creativity in approach, ability to identify meaningful patterns, and quality of insights and recommendations. - 40%
3) Reproducibility: Well-organized code, clear documentation, and transparent workflows. - 5%
4) Visualization & Design: Effective use of charts and visuals to support your conclusions. - 25%
5) Technical Soundness: Appropriate use of statistical methods, reasonable assumptions, and valid interpretations. - 10%
* This will be prioritized more than the experience level of the statistical methods your team members used. This will account for differing experience levels across competitors

## Dataset Acknowledgment
This project utilizes the Motor Vehicle Collisions - Crashes dataset, maintained and publicly released by the City of New York through <catalog.data.gov>. We thank the City of New York for making this valuable public resource freely available for research and analysis.

### Citation (APA 7th edition)
City of New York—Motor Vehicle Collisions—Crashes. (n.d.). Data.Gov. Retrieved October 14, 2026, from http://catalog.data.gov/dataset/motor-vehicle-collisions-crashes

This citation must be included at the end of your code notebook.

## About This Event
Data Challenge 1 was organized by Duke's Statistical Science Majors Union - Competitions and Opportunities Committee. Contributing members of the committee and the SSMU executive team are listed below:
* Sophia Behar, '30
* Hanna Chee, ‘29
* Sabrina Chen, '29
* Anthony Ha, '30
* Hyunjin Lee, ‘27
* Leo Lee, '30
* Susan Li, ‘29
* Phillip Lin, '27
* Liane Ma, ‘27
* Margaret McPheeters, '29
* Chelsea Nguyen, ‘28
* Cooper Ruffing, ‘27
* Keekee Venkatesh, '27
* Derek Wang, ‘28
* Emily Wang, '30
* Amy Xu, ‘27
* And an additional thank-you goes out to the professors who helped advertise the event.
