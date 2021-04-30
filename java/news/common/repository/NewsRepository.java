package news.common.repository;

import common.Repository;
import news.common.model.News;
import org.apache.commons.lang3.reflect.FieldUtils;

import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.reflect.Field;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;

public abstract class NewsRepository<T, U> implements Repository<News> {
    protected final T database;
    protected final String source;

    public NewsRepository(T database, String source) {
        this.database = database;
        this.source = source;
    }

    protected void handleDuplication(News news) {
        warn(new IllegalArgumentException("Duplicate news"), this.getClass(), news.toJSON().toString());
    }

    public abstract long getLastTimestamp();

    public abstract void clearAll(String source);

    protected U toDBObject(News news) {
        return toDBObject(news, toDBFieldMap(news));
    }

    protected Map<String, Field> toDBFieldMap(News news) {
        return FieldUtils.getFieldsListWithAnnotation(news.getClass(), DBField.class)
                .stream()
                .collect(Collectors.toMap(field -> field.getAnnotation(DBField.class).name(), Function.identity()));
    }

    protected abstract U toDBObject(News news, Map<String, Field> dbFieldMap);

    public abstract boolean exists(News news);

    @Retention(RetentionPolicy.RUNTIME)
    public @interface DBField {
        String name();
    }

    public interface EmptinessSearchable<T> {
        Set<T> findEmptyContentNews();
    }
}
