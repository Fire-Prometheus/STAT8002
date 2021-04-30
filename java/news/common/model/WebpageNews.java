package news.common.model;

import common.URLAccessible;
import news.common.repository.NewsRepository;

import java.util.HashSet;
import java.util.Set;

public class
WebpageNews extends News implements URLAccessible, Taggable {
    @NewsRepository.DBField(name = "url")
    protected final String url;
    @NewsRepository.DBField(name = "tags")
    protected final Set<String> tags = new HashSet<>();

    public WebpageNews(String id, String url) {
        super(id);
        this.url = url;
    }

    @Override
    public String getURL() {
        return url;
    }

    @Override
    public void addTag(String tag) {
        tags.add(tag);
    }
}
