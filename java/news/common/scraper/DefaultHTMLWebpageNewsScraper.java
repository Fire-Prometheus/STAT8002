package news.common.scraper;

import news.common.builder.NewsBuilder;
import news.common.builder.WebpageNewsBuilder;
import news.common.model.WebpageNews;
import news.common.repository.NewsRepository;
import org.jsoup.nodes.Element;

import java.io.IOException;
import java.net.URISyntaxException;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

public abstract class DefaultHTMLWebpageNewsScraper<N extends WebpageNews, B extends WebpageNewsBuilder<N, Element, Element, B>> extends HTMLWebpageNewsScraper<N, B> {
    public DefaultHTMLWebpageNewsScraper(boolean isDateSensitiveOnly, NewsRepository<?, ?> repository, int interval, TimeUnit timeUnit) throws IllegalArgumentException {
        super(isDateSensitiveOnly, repository, interval, timeUnit);
    }

    public DefaultHTMLWebpageNewsScraper(NewsRepository<?, ?> repository, int interval, TimeUnit timeUnit) throws IllegalArgumentException {
        super(repository, interval, timeUnit);
    }

    @Override
    protected Set<B> readHeadlines(int page) throws IOException, NoHeadlineFoundException {
        String url = generateURL(page);
        Element body = getBody(url);
        return createNewsBuilders(body);
    }

    protected Set<B> createNewsBuilders(Element body) {
        return body.select(getHeadlineSelector()).stream()
                .map(element -> {
                    try {
                        return createNewsBuilder().with(element);
                    } catch (URISyntaxException | NewsBuilder.NewsBuilderException e) {
                        repository.warn(e, this.getClass(), element.html());
                        return null;
                    } catch (Exception e) {
                        repository.error(e, this.getClass(), element.html());
                        return null;
                    }
                })
                .filter(Objects::nonNull)
                .collect(Collectors.toSet());
    }

    protected abstract String generateURL(int page);
}
