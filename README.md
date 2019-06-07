# SmartReview - making smartphone shopping faster

Website smartcamreview.com is now deactivated since I am not able to keep the app updated as much as I would like. This project was a fun demonstration project that I built over the course of a few weeks as an Insight Data Science fellow. Feel free to message me for a demo though!

Web app to allow users who want to quickly jump to the camera section of YouTube smartphone video reviews, so they don't have to watch the entire video. Saves users on average 9 minutes per video, enabling them to see many camera reviews for a given phone in a much shorter amount of time.



Full process before deployment to web:
  * Download YouTube smartphone video review transcripts using yt_data.py
  * Perform topic modeling using find_topics.py
  * Find start and end times of camera review sections using yt_analyzer.py
  * Tune parameters with yt_validation.py
