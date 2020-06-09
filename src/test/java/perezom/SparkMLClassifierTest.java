package perezom;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.SparkSession;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class SparkMLClassifierTest {

    private SparkSession context;

    @BeforeEach
    void setUp() {
        context = SparkMLClassifier.getSession();
    }

    @Test
    void getSession() {
        assertNotNull(context);
    }

    @Test
    void getData(){
        var data = SparkMLClassifier.getDataset(context);
        assertNotNull(data);
        assertTrue(data instanceof Dataset);
    }

    @Test
    void splitsAreNotNull(){
        var data = SparkMLClassifier.getDataset(context);
        var splits = SparkMLClassifier.getTrainTestSplit(data);
        assertNotNull(splits);
        assertEquals(2, splits.length);
        assertNotNull(splits[0]);
        assertNotNull(splits[1]);
    }
}