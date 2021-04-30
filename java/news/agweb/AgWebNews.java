package news.agweb;

import news.common.model.Categorical;
import news.common.repository.NewsRepository;
import news.common.model.WebpageNews;

import java.util.Set;

public class AgWebNews extends WebpageNews implements Categorical {
    public static final String DOMAIN = "https://www.agweb.com/", SOURCE_NAME = "AgWeb";
    @NewsRepository.DBField(name = "author_title")
    private String authorTitle;

    public AgWebNews(String id, String url, Set<String> categories, String headline) {
        super(id, url);
        this.tags.addAll(categories);
        setTimestamp(timestamp);
        setHeadline(headline);
    }

    @Override
    public Set<String> getCategories() {
        return tags;
    }

    public void setAuthorTitle(String authorTitle) {
        this.authorTitle = authorTitle;
    }
}
