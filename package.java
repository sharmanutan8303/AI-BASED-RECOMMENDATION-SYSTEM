import org.apache.mahout.cf.taste.eval.IRStatistics;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.eval.RecommenderIRStatsEvaluator;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.eval.AverageAbsoluteDifferenceRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.eval.GenericRecommenderIRStatsEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

import java.io.File;
import java.io.IOException;
import java.util.List;

public class RecommendationSystem {
    public static void main(String[] args) throws Exception {
        
        DataModel model = new FileDataModel(new File("data.csv"));
      
        UserSimilarity similarity = new PearsonCorrelationSimilarity(model);

        UserNeighborhood neighborhood = new NearestNUserNeighborhood(2, similarity, model);

        Recommender recommender = new GenericUserBasedRecommender(model, neighborhood, similarity);

        for (LongPrimitiveIterator users = model.getUserIDs(); users.hasNext(); ) {
            long userId = users.nextLong();
            List<RecommendedItem> recommendations = recommender.recommend(userId, 3);
            System.out.printf("User %d recommendations:%n", userId);
            for (RecommendedItem recommendation : recommendations) {
                System.out.printf("Item ID: %d, Value: %.2f%n", recommendation.getItemID(), recommendation.getValue());
            }
        }

        RecommenderEvaluator evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator();
        RecommenderBuilder builder = model1 -> new GenericUserBasedRecommender(model1, neighborhood, similarity);
        double score = evaluator.evaluate(builder, null, model, 0.7, 1.0);
        System.out.printf("Recommender evaluation score: %.4f%n", score);

        RecommenderIRStatsEvaluator statsEvaluator = new GenericRecommenderIRStatsEvaluator();
        IRStatistics stats = statsEvaluator.evaluate(builder, null, model, null, 3, GenericRecommenderIRStatsEvaluator.CHOOSE_THRESHOLD, 1.0);
        System.out.printf("Precision: %.4f, Recall: %.4f%n", stats.getPrecision(), stats.getRecall());
    }
}
