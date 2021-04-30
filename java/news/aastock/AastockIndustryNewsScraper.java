package news.aastock;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonParseException;
import news.common.repository.NewsRepository;
import news.common.scraper.HTMLWebpageNewsScraper;
import org.apache.commons.lang3.StringUtils;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

import java.io.IOException;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;

public class AastockIndustryNewsScraper extends HTMLWebpageNewsScraper<AastockNews, AastockNewsBuilder> {
    private final String industry;

    public AastockIndustryNewsScraper(String sourceName, NewsRepository<?, ?> repository, int interval, TimeUnit timeUnit, String industry) throws IllegalArgumentException {
        super(sourceName, repository, interval, timeUnit);
        this.industry = industry;
    }

    @Override
    protected Set<AastockNewsBuilder> readHeadlines(int page) throws IOException, NoHeadlineFoundException {
        boolean isFirstPage = page <= 1;
        String url = isFirstPage ? "http://www.aastocks.com/tc/stocks/news/aafn-ind/" + industry : "http://www.aastocks.com/tc/resources/datafeed/getmorenews.ashx?cat=ind&sc=" + industry + "&p=" + page;
        Element body = getBody(url);
        Set<AastockNewsBuilder.AastockNewsMetadata> metadataSet;
        if (isFirstPage) {
            Elements headlines = body.select("a[id^=cp_ucAAFNSearch_repNews_lnkNews_]");
            if (headlines.isEmpty()) {
                throw new NoHeadlineFoundException(page);
            }
            metadataSet = headlines.stream()
                    .map(element -> new AastockNewsBuilder.AastockNewsMetadata(element.attr("title"), element.attr("href"), null))
                    .collect(Collectors.toSet());
        } else {
            String json = body.text();
            if (StringUtils.isEmpty(json)) {
                throw new NoHeadlineFoundException(page);
            }
            JsonArray headlines;
            try {
                headlines = new Gson().fromJson(json, JsonArray.class);
            } catch (JsonParseException e) {
                throw new NoHeadlineFoundException(page);
            }
            metadataSet = StreamSupport.stream(headlines.spliterator(), false)
                    .map(JsonElement::getAsJsonObject)
                    .map(jsonObject -> new AastockNewsBuilder.AastockNewsMetadata(jsonObject.getAsJsonPrimitive("h").getAsString(), null, jsonObject.getAsJsonPrimitive("id").getAsString()))
                    .collect(Collectors.toSet());
        }
        return metadataSet.stream()
                .map(aastockNewsMetadata -> {
                    try {
                        return new AastockNewsBuilder(industry).with(aastockNewsMetadata);
                    } catch (Exception e) {
                        repository.error(e, this.getClass(), aastockNewsMetadata.toString());
                        return null;
                    }
                })
                .filter(Objects::nonNull)
                .collect(Collectors.toSet());
    }

    @Override
    protected String getHeadlineSelector() {
        return null;
    }

    @Override
    protected AastockNewsBuilder createNewsBuilder() throws Exception {
        return null;
    }

    @Override
    public void readNews(String url) throws Exception {

    }
}
