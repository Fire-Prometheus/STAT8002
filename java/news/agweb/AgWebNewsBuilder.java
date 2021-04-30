package news.agweb;

import news.common.builder.WebpageNewsBuilder;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

import java.net.URISyntaxException;
import java.text.DateFormat;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.*;

public class AgWebNewsBuilder extends WebpageNewsBuilder<AgWebNews, Element, Element, AgWebNewsBuilder> {
    protected final DateFormat inputDateFormat = createDateFormat("HH:mma MMM dd, yyyy", "EST");
    protected final Set<String> categories;

    public AgWebNewsBuilder(Set<String> categories) throws URISyntaxException {
        super(AgWebNews.DOMAIN);
        this.categories = categories;
    }

    @Override
    public AgWebNewsBuilder with(Element element) throws NewsBuilderException {
        try {
            Element a = element.selectFirst("a.field--name-title");
            String headline = a.text();
            String url = resolve(a.attr("href"));
            news = new AgWebNews(retrieveIDFromURL(url), url, categories, headline);
            Element authorNode = element.selectFirst("field--name-field-author > a");
            if (authorNode != null) {
                String author = authorNode.text();
                String authorTitle = element.selectFirst("span.author-title").text();
                news.setAuthor(author);
//                news.setAuthorTitle(authorTitle);
            }
            return this;
        } catch (Exception e) {
            throw new NewsBuilderException(this.getClass(), e.getMessage(), element.html());
        }
    }

    @Override
    public AgWebNewsBuilder parse(Element element) throws NewsBuilderException {
        try {
            news.setTimestamp(retrieveTimestamp(element));
            Elements paragraphs = element.select("div.node__content div.field--name-body > p");
            news.setContent(paragraphs.text());
//            if (news.getAuthor() == null) {
//                String author = element.selectFirst("div.node-image-caption > span").text();
//                news.setAuthor(author);
//            }
            return this;
        } catch (Exception e) {
            throw new NewsBuilderException(this.getClass(), e.getMessage(), element.html());
        }
    }

    @Override
    public AgWebNewsBuilder url(String url, Element element) throws NewsBuilderException {
        try {
            long epochSecond = retrieveTimestamp(element);
            String headline = element.selectFirst("h1 > span[property=\"schema:name\"]").text();
            Element authorNode = element.selectFirst("article[typeof=\"schema:Person\"] div.field-user--field-full-name div.field-item");
            authorNode = authorNode == null ? element.selectFirst("div.node-image-caption > span") : authorNode;
            String author = authorNode.text();
            news = new AgWebNews(retrieveIDFromURL(url), url, categories, headline);
            news.setTimestamp(epochSecond);
            news.setAuthor(author);
            Elements paragraphs = element.select("div.field-item[property=\"schema:text\"]");
            news.setContent(paragraphs.text());
            return this;
        } catch (Exception e) {
            throw new NewsBuilderException(this.getClass(), e.getMessage(), element.html());
        }
    }

    protected long retrieveTimestamp(Element body) {
        LocalDateTime dateTime = LocalDateTime.parse(body.selectFirst("span[property=\"schema:dateCreated\"]").attr("content"), DateTimeFormatter.ISO_DATE_TIME);
        return dateTime.toEpochSecond(ZoneOffset.ofHours(-5)) * 1000;
    }
}
