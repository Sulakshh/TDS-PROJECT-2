ABOUT GOODREADS.CSV DATASET:- 
Brief about the data I recieved:-
      The dataset consists of 10,000 books with various statistics about their ratings, publication years, and reviews: - 
                Ratings: The average rating for books is 4.00, with ratings ranging from 2.47 to 4.82. The number of ratings varies, with some books receiving just 1 rating, while others get up to 3455 ratings.
                - Reviews: On average, each book has 59,687 ratings and 2,919 text reviews. The number of reviews varies significantly across books. 
                - Publication Year: The books were mostly published around 1981, though the range spans from as early as 1750 to as recent as 2017. 
                - Percentiles: 25% of books have ratings below 3.85, 50% are below 4.02, and 75% are below 4.18. 
                - Book IDs: The dataset includes unique book IDs, Goodreads IDs, and Best Book IDs, with values ranging from 1 to over 33 million. 
      In summary, this dataset provides a detailed overview of 10,000 books, showing a mix of ratings and reviews, with publication years mainly around 1981 but covering a wide range of years. 
      The ratings are mostly positive, with some variation in the number of reviews and ratings per book. Cleaning summary:- The dataset has 10,000 rows, with the following issues: 1. Missing Values: 
                          - 7% missing isbn, 5.85% missing isbn13 and original_title, 0.21% missing original_publication_year, and 10.84% missing language_code. 
                          2. Duplicates: No duplicate rows or book IDs. 
                          3. Invalid Data: 31 rows have negative publication years. 
                          4. Flagged Rows: 2,171 rows need review due to missing values or invalid data. 
      In short, there are missing values in several columns, 31 rows with invalid publication years, but no duplicates. A total of 2,171 rows require further attention.
