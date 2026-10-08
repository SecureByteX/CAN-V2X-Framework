import joblib
a = joblib.load('ml/can/model.pkl')
print('CAN best:', a['model_name'], '| rows:', a['dataset_rows'], '| train:', a['train_rows'], '| test:', a['test_rows'])
for r in a['evaluation_results']:
    print(' ', r['model_name'], 'acc=%.4f' % r['accuracy'], 'macroF1=%.4f' % r['f1_macro'], 'fpr=%.4f' % r['false_positive_rate'], 'infer=%.6f' % r['inference_time_ms'])
print('CAN classes:', a['class_distribution'])
v = joblib.load('ml/v2x/veremi_model.pkl')
print('VEREMI best:', v['model_name'], '| rows:', v['source_rows'], '| train:', v['train_rows'], '| test:', v['test_rows'])
for r in v['results']:
    print(' ', r['model_name'], 'acc=%.4f' % r['accuracy'], 'macroF1=%.4f' % r['f1_macro'], 'fpr=%.4f' % r['false_positive_rate'], 'infer=%.6f' % r['inference_time_ms'])
print('VEREMI nclasses:', len(v['class_distribution']))
import sqlite3
c = sqlite3.connect('database/ids_system.db')
for t in ['can_events','v2x_events','can_alerts','v2x_alerts','model_performance']:
    try:
        print(t, c.execute('SELECT COUNT(*) FROM %s' % t).fetchone()[0])
    except Exception as e:
        print(t, 'ERR', e)
print('can outcomes:', c.execute("SELECT outcome, COUNT(*) FROM can_events GROUP BY outcome").fetchall())
print('v2x outcomes:', c.execute("SELECT outcome, COUNT(*) FROM v2x_events GROUP BY outcome").fetchall())
