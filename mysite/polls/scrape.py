import snscrape.modules.twitter as sntwitter

def get_tweet(username, number_of_tweets):

    tweets = []
    request = 'from:' + username +', -filter:replies'
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(request).get_items()):
        if i > number_of_tweets:
            break
        tweets.append([tweet.content])

    return tweets

# pip3 install git+https://github.com/JustAnotherArchivist/snscrape.git
 
