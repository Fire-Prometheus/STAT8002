package news.agweb;

import news.common.repository.NewsRepository;

import java.util.concurrent.TimeUnit;

public class AgWebBlogNewsScraper extends AgWebNewsScraper {
    public AgWebBlogNewsScraper(NewsRepository<?, ?> repository, int interval, TimeUnit timeUnit) throws IllegalArgumentException {
        super("https://www.agweb.com/blogs", "blog", repository, interval, timeUnit);
    }

    @Override
    protected AgWebNewsBuilder createNewsBuilder() throws Exception {
        return new AgWebBlogNewsBuilder();
    }

    public static void main(String[] args) {
        AgWebBlogNewsScraper agWebBlogNewsScraper = new AgWebBlogNewsScraper(new MongoDBAgWebNewsRepository("blog"), 1, TimeUnit.SECONDS);
        Thread thread = new Thread(agWebBlogNewsScraper);
        thread.start();
    }
}
