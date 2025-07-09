import feedparser
import json
import re
from datetime import datetime
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

categories = ["Artificial Intelligence", "War", "Cyber Security", "Privacy", "Government"]
category_embeddings = model.encode(categories, convert_to_tensor=True)

with open('region_keywords.json', 'r') as f:
    region_keywords = json.load(f)

rss_feeds = [
    {
        "url": "https://haveibeenpwned.com/feed/breaches/",
        "source": "Have I Been Pwned",
        "category": "Data Breaches"
    },
    {
        "url": "https://feeds.feedburner.com/TheHackersNews",
        "source": "The Hacker News"
    },
    {
        "url": "https://www.theguardian.com/world/privacy/rss",
        "source": "The Guardian"
    },
    {
        "url": "https://www.wired.com/feed/category/security/latest/rss",
        "source": "Wired"
    },
    {
        "url": "https://privacyinternational.org/rss.xml",
        "source": "Privacy International"
    }
]

articles = []

date_formats = [
    '%a, %d %b %Y %H:%M:%S %z',
    '%a, %d %b %Y %H:%M:%S',
    '%a, %d %b %Y %H:%M:%S GMT',
]

for feed in rss_feeds:
    feed_url = feed["url"]
    feed_source = feed["source"]
    feed_category = feed.get("category", "")
    
    parsed_feed = feedparser.parse(feed_url)
    for entry in parsed_feed.entries:
        published_date = entry.get('published', '')
        date_object = None

        if published_date:
            for fmt in date_formats:
                try:
                    date_object = datetime.strptime(published_date, fmt)
                    date_object = date_object.replace(tzinfo=None)
                    break
                except ValueError:
                    continue

        article_text = entry.title + " " + (entry.summary if 'summary' in entry else "")

        if feed_category == "":
            article_embedding = model.encode(article_text, convert_to_tensor=True)
            cosine_scores = util.pytorch_cos_sim(article_embedding, category_embeddings)

            for i, score in enumerate(cosine_scores[0]):
                if score > 0.3:
                    feed_category = categories[i]
                    break
                else:
                    feed_category = "News"

        region = "World"
        for r, keywords in region_keywords.items():
            if any(re.search(r'\b' + re.escape(keyword.lower()) + r'\b', article_text.lower()) for keyword in keywords):
                region = r
                break

        article = {
            'title': entry.title,
            'url': entry.link,
            'date': date_object,
            'source': feed_source,
            'category': feed_category,
            'region': region
        }
        articles.append(article)

articles.sort(key=lambda x: x['date'], reverse=True)

for article in articles:
    article['date'] = article['date'].strftime('%d %B %Y') if article['date'] else ''

with open('news.json', 'w') as f:
    json.dump({'articles': articles}, f, indent=4)

# RSS Feed
rss_content = '''<?xml version="1.0" encoding="UTF-8" ?>
<rss version="2.0">
<channel>
    <title>Beginner Privacy News</title>
    <link>http://beginnerprivacy.com/news</link>
    <description>This is an aggregated news feed from various sources.</description>
'''

for article in articles:
    rss_content += f'''
    <item>
        <title>{article['title']}</title>
        <link>{article['url']}</link>
        <pubDate>{article['date']}</pubDate>
        <description>{article['source']} - {article['category']} - {article['region']}</description>
    </item>
'''

rss_content += '''
</channel>
</rss>
'''

with open('news.rss', 'w') as f:
    f.write(rss_content)
