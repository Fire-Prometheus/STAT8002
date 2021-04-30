package news.successfulfarming;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import news.common.repository.NewsRepository;
import news.common.scraper.FirstGetThenPostHTMLWebpageNewsScraper;
import org.apache.commons.lang.StringUtils;
import org.jsoup.Connection;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Element;

import java.util.concurrent.TimeUnit;

public class SuccessfulFarmingNewsScraper extends FirstGetThenPostHTMLWebpageNewsScraper<SuccessfulFarmingNews, SuccessfulFarmingNewsBuilder> {
    private final Category category;

    public SuccessfulFarmingNewsScraper(Category category, NewsRepository<?, ?> repository, int interval, TimeUnit timeUnit) throws IllegalArgumentException {
        super(repository, interval, timeUnit, category.getUrlToGet(), "https://www.agriculture.com/views/ajax");
        this.category = category;
    }

    @Override
    protected String getHeadlineSelector() {
        return ".recent-content-title-teaser";
    }

    @Override
    protected SuccessfulFarmingNewsBuilder createNewsBuilder() throws Exception {
        return new SuccessfulFarmingNewsBuilder(category.getName());
    }

    @Override
    public Element convertResponse(Connection.Response response, int page) throws NoHeadlineFoundException {
        JsonArray jsonArray = new Gson().fromJson(response.body(), JsonArray.class);
        String data = jsonArray.get(1).getAsJsonObject().getAsJsonPrimitive("data").getAsString();
        if (StringUtils.isEmpty(data)) {
            throw new NoHeadlineFoundException(page);
        }
        return Jsoup.parse(data);
    }

    @Override
    public String generateRequestBody(int page) {
        return category.getPostRequestBody(page);
    }

    public enum Category {
        BUSINESS(70, Constants.OFFSET_THREE, "https://www.agriculture.com/news/business"),
        CROPS(71, Constants.OFFSET_THREE, "https://www.agriculture.com/news/crops"),
        TECHNOLOGY(74, Constants.OFFSET_THREE, "https://www.agriculture.com/news/technology"),
        NEWSWIRE(77, Constants.OFFSET_THREE, "https://www.agriculture.com/markets/newswire"),
        COMMODITY_PRICES(78, Constants.DEFAULT, "https://www.agriculture.com/markets/commodity-prices"),
        CROP_MARKET_ANALYSIS(79, Constants.OFFSET_THREE, "https://www.agriculture.com/markets/analysis/crops"),
        YOUR_WORLD_IN_AGRICULTURE(81, Constants.OFFSET_THREE, "https://www.agriculture.com/markets/your-world-in-agriculture"),
        WEATHER(82, Constants.OFFSET_THREE, "https://www.agriculture.com/weather/news"),
        GRAIN_HANDLING(90, Constants.OFFSET_THREE, "https://www.agriculture.com/machinery/grain-handling-and-equipment"),
        HARVESTING(91, Constants.OFFSET_THREE, "https://www.agriculture.com/machinery/harvesting"),
        AGRONOMY_INSIDER(102, Constants.NO_VIDEOS_OFFSET_1, "https://www.agriculture.com/agronomy-insider"),
        CONSERVATION(103, Constants.OFFSET_THREE, "https://www.agriculture.com/crops/conservation"),
        CORN(104, Constants.OFFSET_THREE, "https://www.agriculture.com/crops/corn"),
        FERTILIZERS(106, Constants.OFFSET_THREE, "https://www.agriculture.com/crops/fertilizers"),
        PESTICIDES(107, Constants.OFFSET_THREE, "https://www.agriculture.com/crops/pesticides"),
        SOYBEANS(108, Constants.OFFSET_THREE, "https://www.agriculture.com/crops/soybeans"),
        WHEAT(109, Constants.OFFSET_THREE, "https://www.agriculture.com/crops/wheat"),
        BUSINESS_PLANNING(120, Constants.OFFSET_THREE, "https://www.agriculture.com/farm-management/business-planning"),
        CROP_INSURANCE(121, Constants.OFFSET_THREE, "https://www.agriculture.com/farm-management/crop-insurance"),
        FINANCES(124, Constants.OFFSET_THREE, "https://www.agriculture.com/farm-management/finances-accounting"),
        PROGRAMS_AND_POLICIES(125, Constants.OFFSET_THREE, "https://www.agriculture.com/farm-management/programs-and-policies"),
        SF_BLOG(529, Constants.OFFSET_THREE, "https://www.agriculture.com/news/sf-blog"),
        SUCCESSFUL_MARKETING(550, Constants.NO_VIDEOS_OFFSET_1, "https://www.agriculture.com/markets/successful-marketing"),
        SOIL_HEALTH(695, Constants.NO_VIDEOS_OFFSET_1, "https://www.agriculture.com/crops/soil-health"),
        ;
        private final int categoryNumber;
        private final String viewDisplayID;
        private final String urlToGet;

        Category(int categoryNumber, String viewDisplayID, String baseURL) {
            this.categoryNumber = categoryNumber;
            this.viewDisplayID = viewDisplayID;
            this.urlToGet = baseURL;
        }

        public String getName() {
            return this.name().replaceAll("_", " ").toLowerCase();
        }

        public String getUrlToGet() {
            return urlToGet;
        }

        public String getPostRequestBody(int page) {
            return "view_name=category_content&view_display_id=" + viewDisplayID + "&view_args=" + categoryNumber + "&page=" + page;
        }

        private static class Constants {
            public static final String OFFSET_THREE = "category_recent_content_offset_three";
            public static final String DEFAULT = "category_recent_content";
            public static final String NO_VIDEOS_OFFSET_1 = "recent_content_no_videos_offset_1";
        }
    }

    public static void main(String[] args) {
        Category category = Category.SOIL_HEALTH;
        System.out.println(category.name());
        MongoDBSuccessfulFarmingNewsRepository mongoDBSuccessfulFarmingNewsRepository = new MongoDBSuccessfulFarmingNewsRepository(category);
        SuccessfulFarmingNewsScraper successfulFarmingNewsScraper = new SuccessfulFarmingNewsScraper(category, mongoDBSuccessfulFarmingNewsRepository, 1, TimeUnit.SECONDS);
        Thread thread = new Thread(successfulFarmingNewsScraper);
        thread.start();
    }
}
