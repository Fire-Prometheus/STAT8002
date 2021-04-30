package news.common.repository.firestore;

import com.google.api.core.ApiFuture;
import com.google.auth.oauth2.GoogleCredentials;
import com.google.cloud.firestore.*;
import news.common.model.News;
import news.common.repository.NewsRepository;

import java.lang.reflect.Field;
import java.util.List;
import java.util.Map;
import java.util.Objects;

@Deprecated
public class FirestoreNewsRepository extends NewsRepository<Firestore, Map<String, Object>> {
    private final CollectionReference news;
    private final String source;

    public FirestoreNewsRepository(String source) throws Exception {
        super(
                FirestoreOptions.getDefaultInstance()
                        .toBuilder()
                        .setProjectId("diesel-media-290317")
                        .setCredentials(GoogleCredentials.getApplicationDefault())
                        .build()
                        .getService(),
                source);
        this.source = source;
        news = database.collection("news");
    }

    @Override
    public void save(News news) {
        try {
            ApiFuture<WriteResult> writeResultApiFuture = this.news.document(news.getSource() + "#" + news.getId()).set(toDBObject(news));
            writeResultApiFuture.get();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    public void log(Throwable throwable, Class<?> origin, String additionalMessage, LogSeverity severity) {

    }

    @Override
    public long getLastTimestamp() {
        try {
            Query query = news.whereEqualTo("source", source).orderBy("timestamp", Query.Direction.DESCENDING).limit(1);
            QuerySnapshot queryDocumentSnapshots = query.get().get();
            List<QueryDocumentSnapshot> documents = queryDocumentSnapshots.getDocuments();
            return documents.isEmpty() ? 0 : Objects.requireNonNull(documents.get(0)).getLong("timestamp");
        } catch (Exception e) {
            e.printStackTrace();
            return 0;
        }
    }

    @Override
    public void clearAll(String source) {
        try {
            QuerySnapshot querySnapshot = news.whereEqualTo("source", source).get().get();
            querySnapshot.getDocuments().forEach(queryDocumentSnapshot -> news.document(queryDocumentSnapshot.getId()).delete());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    public Map<String, Object> toDBObject(News news) {
//        Map<String, Object> map = new HashMap<>(7);
//        map.put("id", id);
//        map.put("source", source);
//        map.put("headline", headline);
//        map.put("content", content);
//        map.put("timestamp", timestamp);
//        map.put("author", author);
//        map.put("creationTime", creationTime);
//        return map;
        return null;
    }

    @Override
    protected Map<String, Object> toDBObject(News news, Map<String, Field> dbFieldMap) {
        return null;
    }

    @Override
    public boolean exists(News news) {
        return false;
    }
}
