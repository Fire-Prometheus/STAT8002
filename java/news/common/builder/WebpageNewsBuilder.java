package news.common.builder;

import common.URLAccessible;
import news.common.model.WebpageNews;

import java.net.URI;
import java.net.URISyntaxException;
import java.text.SimpleDateFormat;
import java.util.Collection;
import java.util.TimeZone;

public abstract class WebpageNewsBuilder<N extends WebpageNews, Metadata, Data, ConcreteBuilder extends NewsBuilder<N, Metadata, Data, ConcreteBuilder>> extends NewsBuilder<N, Metadata, Data, ConcreteBuilder> implements URLAccessible {
    protected final URI uri;

    public WebpageNewsBuilder(String domain) throws URISyntaxException {
        this.uri = new URI(domain);
    }

    public abstract ConcreteBuilder url(String url, Data data) throws NewsBuilderException;

    @Override
    public String getURL() {
        return news.getURL();
    }

    protected static SimpleDateFormat createDateFormat(String dateFormat, String zoneID) {
        SimpleDateFormat simpleDateFormat = new SimpleDateFormat(dateFormat);
        simpleDateFormat.setTimeZone(TimeZone.getTimeZone(zoneID));
        return simpleDateFormat;
    }

    protected String resolve(String href) {
        return this.uri.resolve(href).toString();
    }

    protected String combineParagraphs(Collection<String> collection) {
        return String.join("\n", collection);
    }

    protected String retrieveIDFromURL(String url) {
        String trimmedURL = url.replaceFirst("/$", "");
        return trimmedURL.substring(url.lastIndexOf('/') + 1);
    }
}
