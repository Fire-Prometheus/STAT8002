package news.agweb;

import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

import java.net.URISyntaxException;
import java.util.Collections;
import java.util.Date;

public class AgWebBlogNewsBuilder extends AgWebNewsBuilder {
    public AgWebBlogNewsBuilder() throws URISyntaxException {
        super(Collections.singleton("blog"));
    }

    @Override
    public AgWebNewsBuilder with(Element element) throws NewsBuilderException {
        try {
            Date date = inputDateFormat.parse(element.selectFirst("span.content-created").text());
            Element a = element.selectFirst("h2.content-title a");
            String headline = a.text();
            String url = resolve(a.attr("href"));
            news = new AgWebNews(retrieveIDFromURL(url), url, categories, headline);
            news.setTimestamp(date.getTime());
            Element authorNode = element.selectFirst("span.author-name div.field-items");
            if (authorNode != null) {
                String author = authorNode.text();
                String authorTitle = element.selectFirst("span.author-title").text();
                news.setAuthor(author);
                news.setAuthorTitle(authorTitle);
            }
            return this;
        } catch (Exception e) {
            throw new NewsBuilderException(this.getClass(), e.getMessage(), element.html());
        }
    }

    @Override
    public AgWebNewsBuilder parse(Element element) throws NewsBuilderException {
        try {
            Elements paragraphs = element.select(".article-page-body .field-item");
            news.setContent(paragraphs.text());
            Element authorNode;
            if (news.getAuthor() == null && (authorNode = element.selectFirst("div.node-image-caption > span")) != null) {
                news.setAuthor(authorNode.text());
            }
            return this;
        } catch (Exception e) {
            throw new NewsBuilderException(this.getClass(), e.getMessage(), element.html());
        }
    }

    @Override
    public AgWebNewsBuilder url(String url, Element element) throws NewsBuilderException {
        try {
            Date date = inputDateFormat.parse(element.selectFirst(".blog-content-publish-date").ownText());
            String headline = element.selectFirst(".block-pagetitle h1 span").text();
            news = new AgWebNews(retrieveIDFromURL(url), url, categories, headline);
            news.setTimestamp(date.getTime());
            parse(element);
            return this;
        } catch (Exception e) {
            throw new NewsBuilderException(this.getClass(), e.getMessage(), element.html());
        }
    }
}
