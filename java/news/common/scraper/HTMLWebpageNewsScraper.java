package news.common.scraper;

import news.common.builder.WebpageNewsBuilder;
import news.common.model.WebpageNews;
import news.common.repository.NewsRepository;
import org.jsoup.Connection;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;

import java.io.IOException;
import java.util.Objects;
import java.util.SortedSet;
import java.util.TreeSet;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

public abstract class HTMLWebpageNewsScraper<N extends WebpageNews, B extends WebpageNewsBuilder<N, ?, Element, B>> extends NewsScraper<NewsRepository<?, ?>, N, B> {
    public HTMLWebpageNewsScraper(boolean isDateSensitiveOnly, NewsRepository<?, ?> repository, int interval, TimeUnit timeUnit) throws IllegalArgumentException {
        super(isDateSensitiveOnly, repository, interval, timeUnit);
    }

    public HTMLWebpageNewsScraper(String sourceName, boolean isDateSensitiveOnly, NewsRepository<?, ?> repository, int interval, TimeUnit timeUnit) throws IllegalArgumentException {
        super(sourceName, isDateSensitiveOnly, repository, interval, timeUnit);
    }

    public HTMLWebpageNewsScraper(NewsRepository<?, ?> repository, int interval, TimeUnit timeUnit) throws IllegalArgumentException {
        super(repository, interval, timeUnit);
    }

    public HTMLWebpageNewsScraper(String sourceName, NewsRepository<?, ?> repository, int interval, TimeUnit timeUnit) throws IllegalArgumentException {
        super(sourceName, repository, interval, timeUnit);
    }

    @Override
    public SortedSet<N> readNews(int page) throws IOException, NoHeadlineFoundException {
        AtomicInteger atomicInteger = new AtomicInteger(0);
        return readHeadlines(page).stream()
                .map(b -> {
                    sleep(atomicInteger.incrementAndGet());
                    try {
                        return b.parse(get(b.getURL())).build();
                    } catch (Exception e) {
                        repository.error(e, this.getClass(), b.getURL());
                    }
                    return null;
                })
                .filter(Objects::nonNull)
                .collect(Collectors.toCollection(TreeSet::new));
    }

    protected void sleep(int numberOfNews) {
        if (numberOfNews % 5 == 0) {
            try {
                TimeUnit.SECONDS.sleep(5);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }

    protected static Connection getConnection(String url) {
        return Jsoup.connect(url)
                .userAgent(NewsScraper.USER_AGENT)
                .ignoreContentType(true);
    }

    protected Document get(String url) throws IOException {
        return getConnection(url).get();
    }

    protected Element getBody(String url) throws IOException {
        return get(url).body();
    }

    protected abstract String getHeadlineSelector();

    protected abstract B createNewsBuilder() throws Exception;

    public void readNews(String url) throws Exception {
        N news = createNewsBuilder().url(url, get(url)).build();
        repository.save(news);
    }

    protected interface Post<T> {
        default T post(String url, int page) throws IOException {
            Connection.Response response = getConnection(url).requestBody(generateRequestBody(page))
                    .method(Connection.Method.POST)
                    .execute();
            return convertResponse(response, page);
        }

        T convertResponse(Connection.Response response, int page) throws NoHeadlineFoundException;

        String generateRequestBody(int page);
    }
}
