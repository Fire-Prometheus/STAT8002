package common;

public interface Repository<T> {
    enum LogSeverity {
        INFO, DEBUG, WARN, ERROR, FATAL
    }

    void save(T t);

    void log(Throwable throwable, Class<?> origin, String additionalMessage, LogSeverity severity);

    default void info(Class<?> origin, String message) {
        log(null, origin, message, LogSeverity.INFO);
    }

    default void debug(Throwable throwable, Class<?> origin, String additionalMessage) {
        log(throwable, origin, additionalMessage, LogSeverity.DEBUG);
    }

    default void warn(Throwable throwable, Class<?> origin, String additionalMessage) {
        log(throwable, origin, additionalMessage, LogSeverity.WARN);
    }

    default void error(Throwable throwable, Class<?> origin, String additionalMessage) {
        log(throwable, origin, additionalMessage, LogSeverity.ERROR);
    }

    default void fatal(Throwable throwable, Class<?> origin, String additionalMessage) {
        log(throwable, origin, additionalMessage, LogSeverity.FATAL);
    }
}
