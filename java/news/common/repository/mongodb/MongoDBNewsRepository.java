package news.common.repository.mongodb;

import com.mongodb.client.*;
import com.mongodb.client.model.Filters;
import com.mongodb.client.model.Projections;
import com.mongodb.client.model.Sorts;
import news.common.model.News;
import news.common.model.WebpageNews;
import news.common.repository.NewsRepository;
import org.apache.commons.lang.StringUtils;
import org.bson.Document;
import org.bson.conversions.Bson;

import java.lang.reflect.Field;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;

public class MongoDBNewsRepository extends NewsRepository<MongoDatabase, Document> {
    private static final String DATABASE = "STAT8002", COLLECTION = "news";
    protected final MongoCollection<Document> newsCollection, logCollection;
    protected final DateFormat outputDateFormat;

    public MongoDBNewsRepository(String source) {
        super(MongoClients.create().getDatabase(DATABASE), source);
        newsCollection = database.getCollection(COLLECTION);
        logCollection = database.getCollection("log");
        outputDateFormat = new SimpleDateFormat("yyyy-MM-dd");
    }

    @Override
    public long getLastTimestamp() {
        FindIterable<Document> findIterable = newsCollection.find(getSearchCriteria())
                .hintString("source-timestamp-index")
                .sort(Sorts.descending("timestamp"))
                .limit(1);
        Document last = findIterable.first();
        return last == null ? 0 : last.getLong("timestamp");
    }

    protected Bson getSearchCriteria() {
        return Filters.eq("source", source);
    }

    @Override
    public void clearAll(String source) {
        newsCollection.deleteMany(Filters.eq("source", source));
    }

    @Override
    protected Document toDBObject(News news, Map<String, Field> dbFieldMap) {
        Document document = new Document();
        dbFieldMap.remove("id");
        dbFieldMap.forEach((key, field) -> {
            try {
                field.setAccessible(true);
                Object value = field.get(news);
                if (value instanceof Collection<?> && ((Collection<?>) value).isEmpty()) {
                    return;
                }
                document.append(key, value);
            } catch (IllegalAccessException e) {
                error(e, this.getClass(), key);
                error(e, this.getClass(), news.toJSON().toString());
            }
        });
        document.append("_id", generateID(news));
        return document;
    }

    @Override
    public boolean exists(News news) {
        Document document = newsCollection.find(generateExactSearchCriteria(news))
                .limit(1)
                .first();
        return document != null;
    }

    protected Bson generateExactSearchCriteria(News news) {
        return Filters.eq("_id", generateID(news));
    }

    @Override
    public void save(News news) {
        try {
            newsCollection.insertOne(toDBObject(news));
        } catch (Exception e) {
            error(e, this.getClass(), news.toJSON().toString());
        }
    }

    @Override
    public void log(Throwable throwable, Class<?> origin, String additionalMessage, LogSeverity severity) {
        logCollection.insertOne(createLog(throwable, origin, additionalMessage, severity));
    }

    protected Document createLog(Throwable throwable, Class<?> origin, String additionalMessage, LogSeverity severity) {
        Document document = new Document();
        document.append("timestamp", ZonedDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME));
        document.append("severity", severity.name());
        document.append("class", origin.getSimpleName());
        if (throwable != null) {
            document.append("cause", throwable.getLocalizedMessage());
        }
        if (StringUtils.isNotBlank(additionalMessage)) {
            document.append("additional_message", additionalMessage);
        }
        return document;
    }

    protected String generateID(News news) {
        return news.getSource() + "#" + outputDateFormat.format(new Date(news.getTimestamp())) + "#" + news.getId();
    }

    public Set<String> retrieveUnreadURLs(String source) {
        MongoCursor<Document> cursor = logCollection.find(
                Filters.and(
                        StringUtils.isEmpty(source) ? Filters.not(Filters.exists("class")) : Filters.regex("class", "^" + source),
                        Filters.eq("severity", LogSeverity.ERROR.name()),
                        Filters.eq("cause", "Read timed out")
                ))
                .sort(Sorts.ascending("timestamp"))
                .projection(Projections.include("additional_message"))
                .cursor();
        LinkedHashSet<String> results = new LinkedHashSet<>();
        while (cursor.hasNext()) {
            results.add(cursor.next().getString("additional_message"));
        }
        cursor.close();
        return results;
    }
}
