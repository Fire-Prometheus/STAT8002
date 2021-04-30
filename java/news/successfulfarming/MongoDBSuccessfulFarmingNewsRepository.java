package news.successfulfarming;

import news.common.repository.mongodb.MongoDBWebpageNewsRepository;

import java.util.TimeZone;

public class MongoDBSuccessfulFarmingNewsRepository extends MongoDBWebpageNewsRepository {


    public MongoDBSuccessfulFarmingNewsRepository(SuccessfulFarmingNewsScraper.Category category) {
        super(SuccessfulFarmingNews.SOURCE_NAME, category.getName());
        this.outputDateFormat.setTimeZone(TimeZone.getTimeZone("EST"));
    }
}
