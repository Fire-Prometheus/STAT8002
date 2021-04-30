package news.common.model;

import com.google.gson.JsonArray;
import com.google.gson.JsonNull;
import com.google.gson.JsonObject;
import common.Jsonable;
import news.common.repository.NewsRepository;
import org.apache.commons.lang.StringUtils;
import org.apache.commons.lang3.BooleanUtils;
import org.apache.commons.lang3.reflect.FieldUtils;

import java.util.Collection;

public class News implements Comparable<News>, Jsonable<JsonObject> {
    @NewsRepository.DBField(name = "id")
    protected final String id;
    @NewsRepository.DBField(name = "source")
    protected final String source;
    @NewsRepository.DBField(name = "headline")
    protected String headline;
    @NewsRepository.DBField(name = "content")
    protected String content;
    @NewsRepository.DBField(name = "timestamp")
    protected long timestamp;
    @NewsRepository.DBField(name = "author")
    protected String author;
    @NewsRepository.DBField(name = "creation_time")
    protected final long creationTime;

    public News(String id) {
        this.id = id;
        this.source = this.getClass().getSimpleName().replaceFirst("News$", "");
        this.creationTime = System.currentTimeMillis();
    }

    public String getId() {
        return id;
    }

    public String getSource() {
        return source;
    }

    public String getHeadline() {
        return headline;
    }

    public void setHeadline(String headline) {
        this.headline = headline;
    }

    public String getContent() {
        return content;
    }

    public void setContent(String content) {
        this.content = content;
    }

    public void setTimestamp(long timestamp) {
        this.timestamp = timestamp;
    }

    public long getTimestamp() {
        return timestamp;
    }

    public String getAuthor() {
        return author;
    }

    public void setAuthor(String author) {
        this.author = author;
    }

    public long getCreationTime() {
        return creationTime;
    }

    @Override
    public String toString() {
        try {
            return toJSON().toString();
        } catch (Exception e) {
            return "";
        }
    }

    @Override
    public int compareTo(News o) {
        int timeOrder = (int) Math.signum(this.timestamp - o.timestamp);
        return timeOrder == 0 ? this.getId().compareTo(o.getId()) : timeOrder;
    }

    @Override
    public JsonObject toJSON() {
        JsonObject jsonObject = new JsonObject();
        FieldUtils.getAllFieldsList(this.getClass())
                .forEach(field -> {
                    try {
                        Object value = field.get(this);
                        if (value == null) {
                            jsonObject.add(field.getName(), JsonNull.INSTANCE);
                        } else if (value instanceof Number) {
                            jsonObject.addProperty(field.getName(), (Number) value);
                        } else if (value instanceof String) {
                            jsonObject.addProperty(field.getName(), (String) value);
                        } else if (value instanceof Collection<?>) {
                            JsonArray jsonArray = new JsonArray();
                            ((Collection<?>) value).forEach(o -> jsonArray.add(o.toString()));
                            jsonObject.add(field.getName(), jsonArray);
                        } else {
                            throw new UnsupportedOperationException();
                        }
                    } catch (IllegalAccessException | UnsupportedOperationException ignored) {

                    }
                });
        return jsonObject;
    }

    public boolean isValid() {
        return StringUtils.isNotBlank(this.id)
                && StringUtils.isNotBlank(this.content)
                && this.timestamp > 0
                && StringUtils.isNotBlank(this.headline);
    }
}
