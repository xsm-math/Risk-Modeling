-- SQLite: actual six-period UCI histories, ordered earliest -> latest.
-- credit_history is built from observed raw fields; no invented transactions.
WITH ranked AS (
 SELECT *, ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY period_index DESC) AS recency
 FROM credit_history
), aggregate_history AS (
 SELECT customer_id, COUNT(*) AS observed_periods,
   SUM(CASE WHEN status >= 1 THEN 1 ELSE 0 END) AS delayed_months,
   MAX(CASE WHEN status > 0 THEN status ELSE 0 END) AS max_delay,
   AVG(bill_amount) * 1.0 / MAX(credit_limit) AS utilization_mean,
   SUM(payment_amount) * 1.0 /
     (SUM(CASE WHEN bill_amount > 0 THEN bill_amount ELSE 0 END) + 1.0) AS payment_bill_ratio,
   SUM(CASE WHEN recency <= 3 AND status >= 1 THEN 1 ELSE 0 END) AS delayed_months_3,
   SUM(CASE WHEN recency <= 3 THEN payment_amount ELSE 0 END) AS payment_total_3
 FROM ranked GROUP BY customer_id
), latest AS (
 SELECT customer_id, bill_amount * 1.0 / credit_limit AS utilization_latest
 FROM ranked WHERE recency = 1
)
SELECT a.*, l.utilization_latest FROM aggregate_history a
JOIN latest l USING (customer_id) ORDER BY customer_id;
