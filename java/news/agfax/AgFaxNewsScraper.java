package news.agfax;

import news.common.repository.mongodb.MongoDBNewsRepository;
import news.common.repository.NewsRepository;
import news.common.repository.mongodb.MongoDBWebpageNewsRepository;
import news.common.scraper.DefaultHTMLWebpageNewsScraper;

import java.net.SocketTimeoutException;
import java.util.Set;
import java.util.concurrent.TimeUnit;

public class AgFaxNewsScraper extends DefaultHTMLWebpageNewsScraper<AgFaxNews, AgFaxNewsBuilder> {
    public AgFaxNewsScraper(NewsRepository<?, ?> repository, int interval, TimeUnit timeUnit) throws IllegalArgumentException {
        super(true, repository, interval, timeUnit);
    }

    @Override
    protected void sleep(int numberOfNews) {
        if (numberOfNews % 2 == 0) {
            try {
                TimeUnit.SECONDS.sleep(2);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }

    @Override
    protected String generateURL(int page) {
        return AgFaxNews.DOMAIN + "category/a-agfax-news/" + (page <= 1 ? "" : "page/" + page + "/");
    }

    @Override
    protected String getHeadlineSelector() {
        return "div.article-big";
    }

    @Override
    protected AgFaxNewsBuilder createNewsBuilder() throws Exception {
        return new AgFaxNewsBuilder();
    }

    public static void main(String[] args) {
        MongoDBWebpageNewsRepository newsRepository = new MongoDBWebpageNewsRepository(AgFaxNews.SOURCE_NAME);
        AgFaxNewsScraper agFaxNewsScraper = new AgFaxNewsScraper(newsRepository, 10, TimeUnit.SECONDS);
        Set<String> emptyContent = newsRepository.findEmptyContentNews();
        Thread thread = new Thread(agFaxNewsScraper);
        thread.start();
    }
}
