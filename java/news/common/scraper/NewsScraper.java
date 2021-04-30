package news.common.scraper;

import common.Repository;
import news.common.model.News;
import news.common.builder.NewsBuilder;
import news.common.repository.NewsRepository;

import java.io.IOException;
import java.util.Arrays;
import java.util.Set;
import java.util.SortedSet;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

public abstract class NewsScraper<R extends NewsRepository<?, ?>, N extends News, B extends NewsBuilder<N, ?, ?, B>> implements Runnable {
    protected static final String USER_AGENT = "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:80.0) Gecko/20100101 Firefox/80.0";
    protected final String sourceName;
    protected final R repository;
    protected final int interval;
    protected final TimeUnit timeUnit;
    protected long lastTimestamp;
    protected final boolean isDateSensitiveOnly;

    public NewsScraper(String sourceName, boolean isDateSensitiveOnly, R repository, int interval, TimeUnit timeUnit) throws IllegalArgumentException {
        if (interval <= 0) {
            throw new IllegalArgumentException("Invalid time interval. A positive value is expected.");
        }
        this.sourceName = sourceName;
        this.isDateSensitiveOnly = isDateSensitiveOnly;
        this.repository = repository;
        this.interval = interval;
        this.timeUnit = timeUnit;
    }

    public NewsScraper(boolean isDateSensitiveOnly, R repository, int interval, TimeUnit timeUnit) throws IllegalArgumentException {
        if (interval <= 0) {
            throw new IllegalArgumentException("Invalid time interval. A positive value is expected.");
        }
        this.sourceName = this.getClass().getSimpleName().replaceFirst("NewsScraper$", "");
        this.isDateSensitiveOnly = isDateSensitiveOnly;
        this.repository = repository;
        this.interval = interval;
        this.timeUnit = timeUnit;
    }

    public NewsScraper(String sourceName, R repository, int interval, TimeUnit timeUnit) throws IllegalArgumentException {
        this(sourceName, false, repository, interval, timeUnit);
    }

    public NewsScraper(R repository, int interval, TimeUnit timeUnit) throws IllegalArgumentException {
        this(false, repository, interval, timeUnit);
    }

    protected long retrieveLatestTimestamp() {
        return repository.getLastTimestamp();
    }

    abstract protected Set<B> readHeadlines(int page) throws IOException, NoHeadlineFoundException;

    abstract public SortedSet<N> readNews(int page) throws IOException, NoHeadlineFoundException;

    protected long getEarliestTimestamp(SortedSet<N> newsSet) {
        return newsSet.last().getTimestamp();
    }

    protected long getLatestTimestamp(SortedSet<N> newsSet) {
        return newsSet.first().getTimestamp();
    }

    @Override
    public void run() {
        this.lastTimestamp = retrieveLatestTimestamp();
        long currentLatestTimestamp = lastTimestamp;
        int page = 1;
        do {
            try {
                // read news of the current page
                repository.info(this.getClass(), "Reading at page " + page);
                SortedSet<N> newsSet = readNews(page++);
                // compute the timestamp of the latest news we already have
                currentLatestTimestamp = Math.max(currentLatestTimestamp, getLatestTimestamp(newsSet));
                // check whether our database is updated
                repository.info(this.getClass(), "News set - " + newsSet.size() + "\n" + newsSet.toString());
                if (isUpdated(newsSet)) {
                    lastTimestamp = currentLatestTimestamp;
                    repository.info(this.getClass(), "Updated.");
                    page = 1;
                }
                // save the news to the database
                newsSet.stream()
                        .filter(n -> n.getTimestamp() >= lastTimestamp)
                        .forEach(repository::save);
                this.timeUnit.sleep(this.interval);
            } catch (NoHeadlineFoundException e) {
                handleException(e, Repository.LogSeverity.WARN);
                repository.info(this.getClass(), "Read through the news server.");
                page = 1;
            } catch (Exception e) {
                handleException(e, concatStackTrace(e), Repository.LogSeverity.FATAL);
                break;
            }
        } while (true);
    }

    protected boolean isUpdated(SortedSet<N> news) {
        long earliestTimestamp = getEarliestTimestamp(news);
        if (isDateSensitiveOnly) {
            return earliestTimestamp < lastTimestamp || repository.exists(news.first());
        } else {
            return earliestTimestamp < lastTimestamp;
        }
    }

    public static class NoHeadlineFoundException extends RuntimeException {
        public NoHeadlineFoundException(int page) {
            super("No headline is found on page " + page);
        }
    }

    protected void handleException(Exception e, Repository.LogSeverity severity) {
        repository.log(e, this.getClass(), this.getClass().getSimpleName(), severity);
    }

    protected void handleException(Exception e, String additionalMessage, Repository.LogSeverity severity) {
        repository.log(e, this.getClass(), this.getClass().getSimpleName() + "\n" + additionalMessage, severity);
    }

    protected String concatStackTrace(Exception e) {
        return Arrays.stream(e.getStackTrace())
                .map(StackTraceElement::toString)
                .collect(Collectors.joining("\n"));
    }
}
