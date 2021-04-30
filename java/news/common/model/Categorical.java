package news.common.model;

import java.util.Set;

public interface Categorical {
    Set<String> getCategories();

    default String getCategory() {
        return getCategories().iterator().next();
    }
}
