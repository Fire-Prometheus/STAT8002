package news.aastock;

import news.common.model.WebpageNews;

public class AastockNews extends WebpageNews {
    public static final String SOURCE_NAME = "AASTOCK";

    private final String industry;

    public AastockNews(String headline, String url, String id, String industry) {
        super(id, url);
        this.industry = industry;
        setHeadline(headline);
    }
}
