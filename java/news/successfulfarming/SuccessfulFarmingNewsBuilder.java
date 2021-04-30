package news.successfulfarming;

import news.common.builder.WebpageNewsBuilder;
import org.jsoup.nodes.Element;

import java.net.URISyntaxException;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;

public class SuccessfulFarmingNewsBuilder extends WebpageNewsBuilder<SuccessfulFarmingNews, Element, Element, SuccessfulFarmingNewsBuilder> {
    private final String category;


    public SuccessfulFarmingNewsBuilder(String category) throws URISyntaxException {
        super("https://www.agriculture.com/");
        this.category = category;
    }

    @Override
    public SuccessfulFarmingNewsBuilder with(Element element) throws NewsBuilderException {
        try {
            Element a = element.selectFirst("h3 a");
            // url
            String href = a.attr("href"), headline = a.text();
            if (href.startsWith("https://www.agriculture.com/video")) {
                throw new NewsBuilderException(this.getClass(), "Video only", href);
            }
            // subheading
            String subheading = element.selectFirst("div.field-body").text();
            news = new SuccessfulFarmingNews(retrieveIDFromURL(href), href, headline, subheading, category);
            return this;
        } catch (Exception e) {
            throw new NewsBuilderException(this.getClass(), e.getLocalizedMessage(), element.html());
        }
    }

    @Override
    public SuccessfulFarmingNewsBuilder parse(Element element) throws NewsBuilderException {
        try {
            // headline
            String headline = element.selectFirst("h1").text();
            news.setHeadline(headline);
            // author
            Element authorNode = element.selectFirst(".byline-author .field-byline a");
            if (authorNode != null) {
                news.setAuthor(authorNode.ownText());
            }
            // timestamp
            String timestampInISOString = element.selectFirst("[property=\"article:modified_time\"]").attr("content");
            LocalDateTime dateTime = LocalDateTime.parse(timestampInISOString, DateTimeFormatter.ISO_DATE_TIME);
            long epochSecond = dateTime.toEpochSecond(ZoneOffset.ofHours(-5)) * 1000;
            news.setTimestamp(epochSecond);
            // content
            String content = element.selectFirst("div.content div.field-body").text();
            news.setContent(content);
            return this;
        } catch (Exception e) {
            throw new NewsBuilderException(this.getClass(), e.getLocalizedMessage(), element.html());
        }
    }

    @Override
    public SuccessfulFarmingNewsBuilder url(String url, Element element) throws NewsBuilderException {
        try {
            news = new SuccessfulFarmingNews(retrieveIDFromURL(url), url, null, null, category);
            parse(element);
            return this;
        } catch (Exception e) {
            throw new NewsBuilderException(this.getClass(), e.getLocalizedMessage(), element.html());
        }
    }
}
