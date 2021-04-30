package news.agfax;

import news.common.builder.WebpageNewsBuilder;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

import java.net.URISyntaxException;
import java.text.DateFormat;
import java.util.Date;
import java.util.HashSet;
import java.util.Set;

public class AgFaxNewsBuilder extends WebpageNewsBuilder<AgFaxNews, Element, Element, AgFaxNewsBuilder> {
    private static final DateFormat DATE_FORMAT = createDateFormat("MMM dd, yyyy", "EST");
    private final Set<String> urlBlacklist = new HashSet<>();

    public AgFaxNewsBuilder() throws URISyntaxException {
        super(AgFaxNews.DOMAIN);
        urlBlacklist.add("https://agfaxweedsolutions.com");
        urlBlacklist.add("https://agfax.com/WEEKEND/");
    }

    @Override
    public AgFaxNewsBuilder with(Element element) throws NewsBuilderException {
        String href = null;
        try {
            Element archiveTag = element.selectFirst("h2 > a");
            href = archiveTag.attr("href");
            String headline = archiveTag.text();
            if (isBlacklisted(href)) {
                throw new UnsupportedOperationException(href + " is not supported.");
            }
            Element spanTag = element.selectFirst("h2 > span");
            Date date = DATE_FORMAT.parse(spanTag.text());
            news = new AgFaxNews(generateID(href), href, headline, date.getTime());
            return this;
        } catch (Exception e) {
            throw new NewsBuilderException(this.getClass(), e.getMessage(), href);
        }
    }

    @Override
    public AgFaxNewsBuilder parse(Element element) throws NewsBuilderException {
        try {
            Elements paragraphs = element.select("div.entry-content");
            news.setContent(combineParagraphs(paragraphs.eachText()));
            Element byline = element.selectFirst("span.byline");
            byline = byline == null ? element.selectFirst("div.td-post-author-name") : byline;
            news.setAuthor(retrieveSource(byline.text()));
            Elements tagsLinks = element.select("span.tags-links > a");
            if (!tagsLinks.isEmpty()) {
                tagsLinks.eachText().forEach(news::addTag);
            }
            return this;
        } catch (Exception e) {
            throw new NewsBuilderException(this.getClass(), e.getMessage(), element.html());
        }
    }

    @Override
    public AgFaxNewsBuilder url(String url, Element element) throws NewsBuilderException {
        try {
            String headline = element.selectFirst("h1.entry-title").text();
            Date date = DATE_FORMAT.parse(element.selectFirst("time.entry-date").text());
            news = new AgFaxNews(generateID(url), url, headline, date.getTime());
            parse(element);
        } catch (Exception e) {
            throw new NewsBuilderException(this.getClass(), e.getMessage(), element.html());
        }
        return this;
    }

    private String generateID(String url) {
        return url.replaceFirst(AgFaxNews.DOMAIN, "")
                .replaceFirst("/$", "")
                .replaceAll("/", "-");
    }

    private String retrieveSource(String byline) {
        return byline.replaceFirst("^\\w+ ", "");
    }

    private boolean isBlacklisted(String url) {
        return urlBlacklist.stream().anyMatch(url::startsWith);
    }
}
