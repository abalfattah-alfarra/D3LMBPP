# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 20:59:24 2025

@author: ِAL
"""

import time
import tweepy

bearer_token = ''
client = tweepy.Client(bearer_token)

retry_attempts = 5
wait_time = 60  # Start with a 1-minute wait

for attempt in range(retry_attempts):
    try:
        tweets = client.search_recent_tweets(query='Bitcoin', max_results=10)
        for tweet in tweets.data:
            print(tweet.text)
        break
    except tweepy.errors.TooManyRequests:
        print(f"Rate limit exceeded. Waiting for {wait_time} seconds.")
        time.sleep(wait_time)
        wait_time *= 2  # Double the wait time for each retry

