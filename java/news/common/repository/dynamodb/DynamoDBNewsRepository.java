package news.common.repository.dynamodb;

import news.common.model.News;
import news.common.repository.NewsRepository;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.dynamodb.DynamoDbClient;
import software.amazon.awssdk.services.dynamodb.model.*;

import java.lang.reflect.Field;
import java.util.*;

@Deprecated
public class DynamoDBNewsRepository extends NewsRepository<DynamoDbClient, Map<String, AttributeValue>> {
    private static final String TABLE_NAME = "News";
    private final String source;

    public DynamoDBNewsRepository(String source) throws Exception {
        super(
                DynamoDbClient.builder()
                        .region(Region.AP_EAST_1)
                        .build(),
                source);
        this.source = source;
    }

    @Override
    public void save(News news) {
        PutItemRequest putItemRequest = PutItemRequest.builder()
                .tableName(TABLE_NAME)
                .item(toDBObject(news))
                .build();
        try {
            database.putItem(putItemRequest);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    public void log(Throwable throwable, Class<?> origin, String additionalMessage, LogSeverity severity) {

    }

    @Override
    protected void handleDuplication(News news) {
        super.handleDuplication(news);
    }

    @Override
    public long getLastTimestamp() {
        Map<String, Condition> map = new HashMap<>();
        map.put(
                DynamoDBSerializable.Field.SOURCE.getDbFieldName(),
                Condition.builder()
                        .attributeValueList(DynamoDBNewsRepository.DynamoDBSerializable.parseString(source))
                        .comparisonOperator(ComparisonOperator.EQ)
                        .build()
        );
        QueryRequest queryRequest = QueryRequest.builder()
                .tableName(TABLE_NAME)
                .keyConditions(map)
                .scanIndexForward(false)
                .limit(1)
                .build();
        try {
            QueryResponse query = database.query(queryRequest);
            return query.count() == 1 ? Long.parseLong(query.items().get(0).get(DynamoDBSerializable.Field.TIMESTAMP.getDbFieldName()).n()) : 0;
        } catch (Exception e) {
            e.printStackTrace();
            return 0;
        }
    }

    @Override
    public void clearAll(String source) {
        DeleteRequest deleteRequest = DeleteRequest.builder()
                .key(Collections.singletonMap(DynamoDBSerializable.Field.SOURCE.getDbFieldName(), DynamoDBSerializable.parseString(source)))
                .build();
        WriteRequest writeRequest = WriteRequest.builder()
                .deleteRequest(deleteRequest)
                .build();
        BatchWriteItemRequest batchWriteItemRequest = BatchWriteItemRequest.builder()
                .requestItems(Collections.singletonMap(TABLE_NAME, Collections.singletonList(writeRequest)))
                .build();
        database.batchWriteItem(batchWriteItemRequest);
    }

    @Override
    public Map<String, AttributeValue> toDBObject(News news) {
        Map<String, AttributeValue> map = new HashMap<>(7);
//        map.put(DynamoDBSerializable.Field.ID.getDbFieldName(), DynamoDBNewsRepository.DynamoDBSerializable.parseString(id));
//        map.put(DynamoDBSerializable.Field.SOURCE.getDbFieldName(), DynamoDBNewsRepository.DynamoDBSerializable.parseString(source));
//        map.put(DynamoDBSerializable.Field.HEADLINE.getDbFieldName(), DynamoDBNewsRepository.DynamoDBSerializable.parseString(headline));
//        map.put(DynamoDBSerializable.Field.CONTENT.getDbFieldName(), DynamoDBNewsRepository.DynamoDBSerializable.parseString(content));
//        map.put(DynamoDBSerializable.Field.TIMESTAMP.getDbFieldName(), DynamoDBNewsRepository.DynamoDBSerializable.parseNumber(timestamp));
//        map.put(DynamoDBSerializable.Field.AUTHOR.getDbFieldName(), DynamoDBNewsRepository.DynamoDBSerializable.parseString(author));
//        map.put(DynamoDBSerializable.Field.CREATION_TIME.getDbFieldName(), DynamoDBNewsRepository.DynamoDBSerializable.parseNumber(creationTime));
        return map;
    }

    @Override
    protected Map<String, AttributeValue> toDBObject(News news, Map<String, Field> dbFieldMap) {
        return null;
    }

    @Override
    public boolean exists(News news) {
        return false;
    }

    public interface DynamoDBSerializable {
        enum Field {
            ID("id"),
            SOURCE("source"),
            HEADLINE("headline"),
            CONTENT("content"),
            TIMESTAMP("timestamp"),
            AUTHOR("author"),
            CREATION_TIME("creation_time");
            private final String dbFieldName;

            Field(String dbFieldName) {
                this.dbFieldName = dbFieldName;
            }

            public String getDbFieldName() {
                return dbFieldName;
            }
        }

        static AttributeValue parseString(String string) {
            return AttributeValue.builder().s(string).build();
        }

        static AttributeValue parseNumber(Number number) {
            return AttributeValue.builder().n(String.valueOf(number)).build();
        }

        static AttributeValue parseStringCollection(Collection<String> stringCollection) {
            return AttributeValue.builder().ss(stringCollection).build();
        }
    }
}
