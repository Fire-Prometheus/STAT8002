package news.agweb;

import news.common.repository.mongodb.MongoDBWebpageNewsRepository;

import java.util.TimeZone;

public class MongoDBAgWebNewsRepository extends MongoDBWebpageNewsRepository {
    public MongoDBAgWebNewsRepository(String category) {
        super(AgWebNews.SOURCE_NAME, category);
        this.outputDateFormat.setTimeZone(TimeZone.getTimeZone("EST"));
    }
}
