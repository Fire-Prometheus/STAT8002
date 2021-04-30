package news.agfax;

import news.common.model.News;
import news.common.repository.NewsRepository;
import news.common.model.Taggable;
import news.common.model.WebpageNews;

import java.util.HashSet;
import java.util.Set;

public class AgFaxNews extends WebpageNews {
    public static final String DOMAIN = "https://agfax.com/", SOURCE_NAME = "AgFax";

    public AgFaxNews(String id, String url, String headline, long timestamp) {
        super(id, url);
        setHeadline(headline);
        setTimestamp(timestamp);
    }
}
