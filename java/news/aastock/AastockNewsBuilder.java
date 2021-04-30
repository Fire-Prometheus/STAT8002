package news.aastock;

import com.google.gson.JsonObject;
import news.common.builder.WebpageNewsBuilder;
import org.jsoup.nodes.Element;

import java.net.URISyntaxException;
import java.text.DateFormat;
import java.text.SimpleDateFormat;

public class AastockNewsBuilder extends WebpageNewsBuilder<AastockNews, AastockNewsBuilder.AastockNewsMetadata, Element, AastockNewsBuilder> {
    private static final DateFormat DATE_FORMAT = new SimpleDateFormat("yyyy/MM/dd HH:mm");
    private final String industry;

    public AastockNewsBuilder(String industry) throws URISyntaxException {
        super("http://www.aastocks.com");
        this.industry = industry;
    }

    @Override
    public AastockNewsBuilder with(AastockNewsMetadata aastockNewsMetadata) throws NewsBuilderException {
        try {
            String title = aastockNewsMetadata.headline;
            String href = this.resolve(aastockNewsMetadata.hasURL() ? aastockNewsMetadata.url : generateURL(aastockNewsMetadata.id));
            String minimizedURL = minimizeURL(href);
            String id = aastockNewsMetadata.hasURL() ? extractID(minimizedURL) : aastockNewsMetadata.id;
            news = new AastockNews(title, minimizedURL, id, industry);
            return this;
        } catch (Exception e) {
            throw new NewsBuilderException(this.getClass(), e.getMessage());
        }
    }

    @Override
    public AastockNewsBuilder parse(Element element) throws NewsBuilderException {
        try {
            news.setContent(element.selectFirst("div#spanContent p").text());
            Element timestamp = element.selectFirst(".newstime5");
            news.setTimestamp(DATE_FORMAT.parse(timestamp.text()).getTime());
            return this;
        } catch (Exception e) {
            JsonObject debugJSON = new JsonObject();
            debugJSON.addProperty("id", news.getId());
            debugJSON.addProperty("document", element.toString());
            throw new NewsBuilderException(this.getClass(), e.getMessage(), debugJSON.toString());
        }
    }

    @Override
    public AastockNewsBuilder url(String url, Element element) throws NewsBuilderException {
        return null;
    }

    private String minimizeURL(String url) {
        return url.endsWith("/industry-news") ? url.replaceFirst("/industry-news$", "") : url;
    }

    private String extractID(String url) {
        return url.substring(url.lastIndexOf('/') + 1);
    }

    private String generateURL(String id) {
        return resolve("/tc/stocks/news/aafn-con/" + id);
    }

    static class AastockNewsMetadata {
        private final String headline, url, id;

        public AastockNewsMetadata(String headline, String url, String id) {
            this.headline = headline;
            this.url = url;
            this.id = id;
        }

        private boolean hasURL() {
            return this.url != null;
        }

        @Override
        public String toString() {
            return "AastockNewsMetadata{" +
                    "headline='" + headline + '\'' +
                    ", url='" + url + '\'' +
                    ", id='" + id + '\'' +
                    '}';
        }
    }
}
