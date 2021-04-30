package common;

public interface Jsonable<JSON> {
    JSON toJSON() throws Exception;
}
