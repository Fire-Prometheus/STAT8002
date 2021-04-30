package news.common.repository.mongodb;

import com.mongodb.client.MongoCursor;
import com.mongodb.client.model.Filters;
import com.mongodb.client.model.Updates;
import news.common.model.Categorical;
import news.common.model.News;
import news.common.repository.NewsRepository;
import org.bson.Document;
import org.bson.conversions.Bson;

import java.util.HashSet;
import java.util.Set;

public class MongoDBWebpageNewsRepository extends MongoDBNewsRepository implements NewsRepository.EmptinessSearchable<String> {
    private final String category;

    public MongoDBWebpageNewsRepository(String source) {
        super(source);
        this.category = null;
    }

    public MongoDBWebpageNewsRepository(String source, String category) {
        super(source);
        this.category = category;
    }

    @Override
    public void save(News news) {
        try {
            if (exists(news)) {
                if (category == null) {
                    newsCollection.findOneAndUpdate(Filters.and(getExactFilter(news), Filters.eq("content", "")), Updates.set("content", news.getContent()));
                } else {
                    newsCollection.findOneAndUpdate(getExactFilter(news), Updates.addToSet("tags", ((Categorical) news).getCategory()));
                }
            } else {
                newsCollection.insertOne(toDBObject(news));
            }
        } catch (Exception e) {
            error(e, this.getClass(), news.toJSON().toString());
        }
    }

    protected Bson getExactFilter(News news) {
        return Filters.eq("_id", generateID(news));
    }

    @Override
    protected Bson getSearchCriteria() {
        return category == null
                ? super.getSearchCriteria()
                : Filters.and(
                super.getSearchCriteria(),
                Filters.in("tags", category));
    }

    @Override
    protected Document createLog(Throwable throwable, Class<?> origin, String additionalMessage, LogSeverity severity) {
        Document log = super.createLog(throwable, origin, additionalMessage, severity);
        log.append("category", category);
        return log;
    }

    @Override
    public Set<String> findEmptyContentNews() {
        MongoCursor<Document> cursor = newsCollection.find(Filters.and(
                getSearchCriteria(),
                Filters.eq("content", "")
        )).cursor();
        Set<String> newsSet = new HashSet<>();
        while (cursor.hasNext()) {
            Document next = cursor.next();
            newsSet.add(next.getString("url"));
        }
        return newsSet;
    }
}
