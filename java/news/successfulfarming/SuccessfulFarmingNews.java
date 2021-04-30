package news.successfulfarming;

import com.google.gson.JsonObject;
import news.common.model.Categorical;
import news.common.model.WebpageNews;

import java.util.Collections;
import java.util.Set;

public class SuccessfulFarmingNews extends WebpageNews implements Categorical {
    public static final String SOURCE_NAME = "SuccessfulFarming";

    public SuccessfulFarmingNews(String id, String url, String headline, String subheading, String category) {
        super(id, url);
        setHeadline(headline);
        JsonObject jsonObject = new JsonObject();
        jsonObject.addProperty("subheading", subheading);
        this.tags.addAll(Collections.singleton(category));
    }

    @Override
    public Set<String> getCategories() {
        return tags;
    }
}
