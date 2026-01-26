# Knowledge Base Inputs

Add article URLs or video links here for processing.

## Input Format

### Articles (URLs)
Create a text file with one URL per line:
```
https://www.mql5.com/en/articles/123
https://www.mql5.com/en/articles/456
```

### Videos (YouTube, etc.)
Add URLs with `#video` prefix:
```
#video https://www.youtube.com/watch?v=xxxxx
```

## Processing

```bash
# Process articles
python scripts/firecrawl_scraper.py --input data/inputs/articles.txt

# Process videos (NPRD)
python tools/nprd/process_video.py --input data/inputs/videos.txt
```
