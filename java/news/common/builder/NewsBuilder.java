package news.common.builder;

import news.common.model.News;
import org.apache.commons.lang3.StringUtils;

public abstract class NewsBuilder<N extends News, Metadata, Data, ConcreteBuilder extends NewsBuilder<N, Metadata, Data, ConcreteBuilder>> {
    protected N news;

    public abstract ConcreteBuilder with(Metadata metadata) throws NewsBuilderException;

    public abstract ConcreteBuilder parse(Data data) throws NewsBuilderException;

    public N build() throws NewsBuilderException {
        if (!news.isValid()) {
            throw new NewsBuilderException((Class<? extends NewsBuilder<? extends News, ?, ?, ?>>) this.getClass(), "Invalid news", news.toJSON().toString());
        }
        return news;
    }

    public static class NewsBuilderException extends RuntimeException {
        public NewsBuilderException(Class<? extends NewsBuilder<? extends News, ?, ?, ?>> newsBuilderClass, String message, String additionalMessage) {
            super(newsBuilderClass.getSimpleName() + "\t" + message + (StringUtils.isEmpty(additionalMessage) ? "" : "\t" + additionalMessage));
        }

        public NewsBuilderException(Class<? extends NewsBuilder<? extends News, ?, ?, ?>> newsFactoryClass, String message) {
            this(newsFactoryClass, message, null);
        }
    }
}
