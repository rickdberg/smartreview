# SmartReview - making smartphone shopping faster

Visit smartcamreview.com to see it in action!

Web app to allow users who want to quickly jump to the camera section of YouTube smartphone video reviews, so they don't have to watch the entire video. Saves users on average 9 minutes per video, enabling them to see many camera reviews for a given phone in a much shorter amount of time.



Full process before deployment to web:
  * Download YouTube smartphone video review transcripts using yt_data.py
  * Perform topic modeling using find_topics.py
  * Find start and end times of camera review sections using yt_analyzer.py
  * Tune parameters with yt_validation.py
