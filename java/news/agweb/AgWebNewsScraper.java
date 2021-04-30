package news.agweb;

import news.common.repository.NewsRepository;
import news.common.scraper.DefaultHTMLWebpageNewsScraper;

import java.util.Collections;
import java.util.Set;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.stream.Stream;

public class AgWebNewsScraper extends DefaultHTMLWebpageNewsScraper<AgWebNews, AgWebNewsBuilder> {
    private final String category, baseURL;

    public AgWebNewsScraper(String baseURL, String category, NewsRepository<?, ?> repository, int interval, TimeUnit timeUnit) throws IllegalArgumentException {
        super(repository, interval, timeUnit);
        this.baseURL = baseURL;
        this.category = category;
    }

    @Override
    protected String generateURL(int page) {
        return baseURL + "?page=" + (page - 1);
    }

    @Override
    protected String getHeadlineSelector() {
        return "div.teaser";
    }

    @Override
    protected AgWebNewsBuilder createNewsBuilder() throws Exception {
        return new AgWebNewsBuilder(Collections.singleton(category));
    }

    public static void main(String[] args) {
        ScheduledExecutorService scheduledExecutorService = Executors.newScheduledThreadPool(12);
        int interval = 2;
        Stream.of(
//                new AgWebNewsScraper("https://www.agweb.com/topics/policy", "policy", new MongoDBAgWebNewsRepository("policy"), interval, TimeUnit.SECONDS),
                new AgWebNewsScraper("https://www.agweb.com/topics/markets", "market", new MongoDBAgWebNewsRepository("market"), interval, TimeUnit.SECONDS)
//                new AgWebNewsScraper("https://www.agweb.com/markets/topics/usda", "USDA", new MongoDBAgWebNewsRepository("USDA report"), interval, TimeUnit.SECONDS)
//                new AgWebNewsScraper("https://www.agweb.com/markets/pro-farmer-analysis", "pro farmer analysis", new MongoDBAgWebNewsRepository("pro farmer analysis"), interval, TimeUnit.SECONDS),
//                new AgWebNewsScraper("https://www.agweb.com/weather/news", "weather", new MongoDBAgWebNewsRepository("weather"), interval, TimeUnit.SECONDS),
//                new AgWebNewsScraper("https://www.agweb.com/crops/harvest-news-and-updates", "harvest", new MongoDBAgWebNewsRepository("harvest"), interval, TimeUnit.SECONDS),
//                new AgWebNewsScraper("https://www.agweb.com/crops/corn", "corn", new MongoDBAgWebNewsRepository("corn"), interval, TimeUnit.SECONDS),
//                new AgWebNewsScraper("https://www.agweb.com/crops/soybeans", "soybean", new MongoDBAgWebNewsRepository("soybean"), interval, TimeUnit.SECONDS),
//                new AgWebNewsScraper("https://www.agweb.com/crops/other-crops", "other crops", new MongoDBAgWebNewsRepository("other crops"), interval, TimeUnit.SECONDS),
//                new AgWebNewsScraper("https://www.agweb.com/farm-business/agribusiness", "business", new MongoDBAgWebNewsRepository("business"), interval, TimeUnit.SECONDS),
//                new AgWebNewsScraper("https://www.agweb.com/farmland", "farmland", new MongoDBAgWebNewsRepository("farmland"), interval, TimeUnit.SECONDS),
//                new AgWebNewsScraper("https://www.agweb.com/technology", "technology", new MongoDBAgWebNewsRepository("technology"), interval, TimeUnit.SECONDS)
        )
                .forEach(agWebNewsScraper -> scheduledExecutorService.schedule(agWebNewsScraper, 10, TimeUnit.SECONDS));
//        MongoDBAgWebNewsRepository mongoDBAgWebNewsRepository = new MongoDBAgWebNewsRepository(null);
//        AgWebNewsScraper agWebNewsScraper = new AgWebNewsScraper(null, null, mongoDBAgWebNewsRepository, 1, TimeUnit.SECONDS);
//        Set<String> emptyContentNews = mongoDBAgWebNewsRepository.findEmptyContentNews();
//        emptyContentNews.forEach(s -> {
//            try {
//                agWebNewsScraper.readNews(s);
//                TimeUnit.SECONDS.sleep(5);
//            } catch (Exception e) {
//                System.out.println(s);
//                e.printStackTrace();
//            }
//        });
    }
}
