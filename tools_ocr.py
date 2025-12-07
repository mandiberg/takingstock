import os
import re

class OCRTools:
    """
    Two-Phase Iterative TSP Solver with Point Skipping
    
    Phase 1: Solve TSP for current point set
    Phase 2: Greedily remove high-overhead points
    Iterate: Re-optimize TSP on remaining points
    """
    
    def __init__(self, DEBUGGING=False):
        self.DEBUGGING = DEBUGGING      

    def ocr_on_cropped_image(self,cropped_image_bbox, ocr, image_filename):
        result = ocr.predict(cropped_image_bbox)
        for res in result:
            # Extract polygons and recognized texts from the OCR result
            dt_polys = res.get('dt_polys') or []
            found_texts = res.get('rec_texts') or []
            good_texts = []
            # Debug: show counts to ensure zip will iterate
            # print(f"dt_polys count: {len(dt_polys)}, rec_texts count: {len(found_texts)}")

            # Iterate pairs; if lengths mismatch, iterate by index to avoid empty zip
            pair_count = min(len(dt_polys), len(found_texts))
            for i in range(pair_count):
                poly = dt_polys[i]
                text = found_texts[i]
                # print(f"Detected text: {text}")
                angle = (poly[1][1] - poly[0][1]) / (poly[1][0] - poly[0][0] + 1e-6)
                # print(f"Detected text: {text} with angle: {angle}")
                if abs(angle) < 0.5:  # filter out highly tilted text
                    good_texts.append(text)
            if self.DEBUGGING: res.save_to_img(os.path.join("output", image_filename))
        return good_texts


    def assign_slogan_id(self, session, openai_client, Slogans, slogan_dict, image_filename, found_texts):
        slogan_id = None
        if bool(found_texts) is False:
            print(f"No text found in image {image_filename}, will assign blank = True.")
            slogan_id = 1 # blank sign - no slogan
        else:
            slogan_id = self.check_existing_slogans(found_texts, slogan_dict)
            if slogan_id is None:
                refined_text = self.clean_ocr_text(openai_client, found_texts)
                print("No match, so refined Text:", refined_text)
                slogan_id = self.check_existing_slogans(refined_text, slogan_dict)

            if slogan_id is not None:
                print(f"Slogan already exists in DB with slogan_id: {slogan_id}.")
            else:
                # Save new slogan to DB
                slogan_id = self.save_slogan_text(session, Slogans, refined_text)
                slogan_dict[slogan_id] = refined_text
        return slogan_id

    def normalize(self, s):
        s = s.upper()
        s = re.sub(r"[^A-Z0-9\s!?'.,-]", "", s)
        s = re.sub(r"\s+", " ", s)
        return s.strip()

    # def correct_words(text):
    #     corrected = []
    #     for word in text.split():
    #         fixed = spell.correction(word)
    #         corrected.append(fixed.upper())
    #     return " ".join(corrected)

    def refine_phrase(self, openai_client, phrase):
        prompt = f"Correct this noisy OCR text into a meaningful English phrase:\n'{phrase}'\nCorrected:"
        response = openai_client.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": prompt}]
        )
        corrected_text = response.choices[0].message.content.strip()
        slogan_text = corrected_text.replace('Corrected: ', '')
        return slogan_text
        

    def clean_ocr_text(self, openai_client, tokens):
        if not tokens:
            return None
        # tokens = ['WOMAN','OWER'] etc.
        raw = " ".join(tokens)
        raw = self.normalize(raw)
        # Step 1: Basic spell correction of individual words
        # word_fixed = correct_words(raw)
        # print("Spell Corrected Text:", word_fixed)
        # Step 2: Language model refinement (contextual)
        final = self.refine_phrase(openai_client, raw)    
        return final

    def get_all_slogans(self, session, Slogans):
        slogans = session.query(Slogans).all()
        slogan_dict = {slogan.slogan_id: slogan.slogan_text for slogan in slogans}
        return slogan_dict

    def check_existing_slogans(self, found_texts, slogan_dict):
        if isinstance(found_texts, list):
            text = " ".join(found_texts)
        else:
            text = found_texts
        normalized_text = self.normalize(text)
        for slogan_id, slogan_text in slogan_dict.items():
            if normalized_text == self.normalize(slogan_text):
                return slogan_id
        return None
    
    def save_slogan_text(self, session, Slogans, slogan_text):
        new_slogan = Slogans(slogan_text=slogan_text)
        session.add(new_slogan)
        session.commit()
        return new_slogan.slogan_id
    
    def save_images_slogans(self, session, ImagesSlogans, image_filename, slogan_id):
        new_placard = ImagesSlogans(image_filename=image_filename, slogan_id=slogan_id)
        session.add(new_placard)
        session.commit()
