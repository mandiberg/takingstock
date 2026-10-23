-- Export the canonical ObjectSignatures registry in the exact CSV shape used by the project:
-- cluster_id,ccount,LH,RH,TF,LE,RE,MO,SH,WA,FT
--
-- The table stores each signature as a token string like:
--   LH:67|RH:67|TF:0|LE:0|RE:0|MO:0|SH:0|WA:0|FT:0
-- The query below parses that token into slot columns and counts how many images map to each cluster_id.
--
-- Use this as a reusable SELECT, or uncomment the INTO OUTFILE line to write a CSV snapshot.

SELECT
    os.cluster_id AS cluster_id,
    COUNT(ios.image_id) AS ccount,
    CAST(
        IFNULL(
            NULLIF(
                TRIM(LEADING 'LH:' FROM SUBSTRING_INDEX(SUBSTRING_INDEX(os.slot_signature_token, 'LH:', -1), '|', 1)),
                ''
            ),
            0
        ) AS SIGNED
    ) AS LH,
    CAST(
        IFNULL(
            NULLIF(
                TRIM(LEADING 'RH:' FROM SUBSTRING_INDEX(SUBSTRING_INDEX(os.slot_signature_token, 'RH:', -1), '|', 1)),
                ''
            ),
            0
        ) AS SIGNED
    ) AS RH,
    CAST(
        IFNULL(
            NULLIF(
                TRIM(LEADING 'TF:' FROM SUBSTRING_INDEX(SUBSTRING_INDEX(os.slot_signature_token, 'TF:', -1), '|', 1)),
                ''
            ),
            0
        ) AS SIGNED
    ) AS TF,
    CAST(
        IFNULL(
            NULLIF(
                TRIM(LEADING 'LE:' FROM SUBSTRING_INDEX(SUBSTRING_INDEX(os.slot_signature_token, 'LE:', -1), '|', 1)),
                ''
            ),
            0
        ) AS SIGNED
    ) AS LE,
    CAST(
        IFNULL(
            NULLIF(
                TRIM(LEADING 'RE:' FROM SUBSTRING_INDEX(SUBSTRING_INDEX(os.slot_signature_token, 'RE:', -1), '|', 1)),
                ''
            ),
            0
        ) AS SIGNED
    ) AS RE,
    CAST(
        IFNULL(
            NULLIF(
                TRIM(LEADING 'MO:' FROM SUBSTRING_INDEX(SUBSTRING_INDEX(os.slot_signature_token, 'MO:', -1), '|', 1)),
                ''
            ),
            0
        ) AS SIGNED
    ) AS MO,
    CAST(
        IFNULL(
            NULLIF(
                TRIM(LEADING 'SH:' FROM SUBSTRING_INDEX(SUBSTRING_INDEX(os.slot_signature_token, 'SH:', -1), '|', 1)),
                ''
            ),
            0
        ) AS SIGNED
    ) AS SH,
    CAST(
        IFNULL(
            NULLIF(
                TRIM(LEADING 'WA:' FROM SUBSTRING_INDEX(SUBSTRING_INDEX(os.slot_signature_token, 'WA:', -1), '|', 1)),
                ''
            ),
            0
        ) AS SIGNED
    ) AS WA,
    CAST(
        IFNULL(
            NULLIF(
                TRIM(LEADING 'FT:' FROM SUBSTRING_INDEX(SUBSTRING_INDEX(os.slot_signature_token, 'FT:', -1), '|', 1)),
                ''
            ),
            0
        ) AS SIGNED
    ) AS FT
FROM ObjectSignatures os
LEFT JOIN ImagesObjectSignatures ios
    ON ios.cluster_id = os.cluster_id
GROUP BY
    os.cluster_id,
    os.slot_signature_token
ORDER BY
    os.cluster_id;

-- To emit a CSV snapshot matching the old export layout, uncomment and update the target path below:
-- SELECT
--     os.cluster_id AS cluster_id,
--     COUNT(ios.image_id) AS ccount,
--     CAST(IFNULL(NULLIF(TRIM(LEADING 'LH:' FROM SUBSTRING_INDEX(SUBSTRING_INDEX(os.slot_signature_token, 'LH:', -1), '|', 1)), ''), 0) AS SIGNED) AS LH,
--     CAST(IFNULL(NULLIF(TRIM(LEADING 'RH:' FROM SUBSTRING_INDEX(SUBSTRING_INDEX(os.slot_signature_token, 'RH:', -1), '|', 1)), ''), 0) AS SIGNED) AS RH,
--     CAST(IFNULL(NULLIF(TRIM(LEADING 'TF:' FROM SUBSTRING_INDEX(SUBSTRING_INDEX(os.slot_signature_token, 'TF:', -1), '|', 1)), ''), 0) AS SIGNED) AS TF,
--     CAST(IFNULL(NULLIF(TRIM(LEADING 'LE:' FROM SUBSTRING_INDEX(SUBSTRING_INDEX(os.slot_signature_token, 'LE:', -1), '|', 1)), ''), 0) AS SIGNED) AS LE,
--     CAST(IFNULL(NULLIF(TRIM(LEADING 'RE:' FROM SUBSTRING_INDEX(SUBSTRING_INDEX(os.slot_signature_token, 'RE:', -1), '|', 1)), ''), 0) AS SIGNED) AS RE,
--     CAST(IFNULL(NULLIF(TRIM(LEADING 'MO:' FROM SUBSTRING_INDEX(SUBSTRING_INDEX(os.slot_signature_token, 'MO:', -1), '|', 1)), ''), 0) AS SIGNED) AS MO,
--     CAST(IFNULL(NULLIF(TRIM(LEADING 'SH:' FROM SUBSTRING_INDEX(SUBSTRING_INDEX(os.slot_signature_token, 'SH:', -1), '|', 1)), ''), 0) AS SIGNED) AS SH,
--     CAST(IFNULL(NULLIF(TRIM(LEADING 'WA:' FROM SUBSTRING_INDEX(SUBSTRING_INDEX(os.slot_signature_token, 'WA:', -1), '|', 1)), ''), 0) AS SIGNED) AS WA,
--     CAST(IFNULL(NULLIF(TRIM(LEADING 'FT:' FROM SUBSTRING_INDEX(SUBSTRING_INDEX(os.slot_signature_token, 'FT:', -1), '|', 1)), ''), 0) AS SIGNED) AS FT
-- FROM ObjectSignatures os
-- LEFT JOIN ImagesObjectSignatures ios ON ios.cluster_id = os.cluster_id
-- GROUP BY os.cluster_id, os.slot_signature_token
-- ORDER BY os.cluster_id
-- INTO OUTFILE '/Users/michaelmandiberg/Documents/GitHub/takingstock/utilities/data/ImagesObjectSignatures_ObjectSignatures_202604272237.csv'
-- FIELDS TERMINATED BY ','
-- OPTIONALLY ENCLOSED BY '"'
-- LINES TERMINATED BY '\n';
