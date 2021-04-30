package news.common.scraper;

import news.common.builder.WebpageNewsBuilder;
import news.common.model.WebpageNews;
import news.common.repository.NewsRepository;
import org.jsoup.nodes.Element;

import java.io.IOException;
import java.util.Set;
import java.util.concurrent.TimeUnit;

public abstract class FirstGetThenPostHTMLWebpageNewsScraper<N extends WebpageNews, B extends WebpageNewsBuilder<N, Element, Element, B>> extends DefaultHTMLWebpageNewsScraper<N, B> implements HTMLWebpageNewsScraper.Post<Element> {
    protected final String urlToGet, urlToPost;

    public FirstGetThenPostHTMLWebpageNewsScraper(boolean isDateSensitiveOnly, NewsRepository<?, ?> repository, int interval, TimeUnit timeUnit, String urlToGet, String urlToPost) throws IllegalArgumentException {
        super(isDateSensitiveOnly, repository, interval, timeUnit);
        this.urlToGet = urlToGet;
        this.urlToPost = urlToPost;
    }

    public FirstGetThenPostHTMLWebpageNewsScraper(NewsRepository<?, ?> repository, int interval, TimeUnit timeUnit, String urlToGet, String urlToPost) throws IllegalArgumentException {
        super(repository, interval, timeUnit);
        this.urlToGet = urlToGet;
        this.urlToPost = urlToPost;
    }

    @Override
    protected Set<B> readHeadlines(int page) throws IOException, NoHeadlineFoundException {
        String url = generateURL(page);
        Element body = page <= 1 ? getBody(url) : post(url, page);
        return createNewsBuilders(body);
    }

    @Override
    protected String generateURL(int page) {
        return page <= 1 ? urlToGet : urlToPost;
    }
}
